import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.components.KRP import KernelResidualProjector
from src.models.components.Lyra_encoder import Lyra
from src.models.components.fusion import Hla2pepAttentionFusion
from src.models.components.predictors import SequencePredictor
from src.models.components.star_fusion import StarFusionHub


def weight_initial(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            if m.weight is not None:
                nn.init.constant_(m.weight.data, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()


class ViewProjector(nn.Module):
    def __init__(self, target_dim):
        super().__init__()

        self.VIEW_DIMS = [20, 11, 23, 24]
        self.num_views = len(self.VIEW_DIMS)

        self.projections = nn.ModuleList([
            nn.Linear(dim, target_dim) for dim in self.VIEW_DIMS
        ])
        self.norm = nn.LayerNorm(target_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, view_list):
        projected_views = []
        for proj_layer, view in zip(self.projections, view_list):
            # 确保输入是 [Batch, Dim, Len] -> 转置为 [Batch, Len, Dim] 进行线性映射
            view = view.transpose(-1, -2)
            x = proj_layer(view)
            x = self.dropout(self.activation(self.norm(x)))
            projected_views.append(x)
        return projected_views


class StarLyraMHC(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(StarLyraMHC, self).__init__()
        self.cfg = cfg
        self.d_model = cfg.model.params.d_model
        dropout = cfg.model.params.dropout

        # 1. 视图投影器 (保留，用于将不同维度的特征对齐到 d_model)
        self.view_projector = ViewProjector(self.d_model)
        self.num_views = self.view_projector.num_views  # 4

        # 魏征修正：废除 KernelResidualProjector
        # 改为：为每个视图配备独立的 Lyra 编码器
        # 参数量警告：这里会实例化 4 个 Lyra，显存占用 x4
        self.hla_encoders = nn.ModuleList([
            Lyra(**cfg.model.encoder_hla) for _ in range(self.num_views)
        ])

        self.pep_encoders = nn.ModuleList([
            Lyra(**cfg.model.encoder_pep) for _ in range(self.num_views)
        ])

        # StarHub 保持不变，它负责聚合 4 个独立专家的意见
        self.hla_hub = StarFusionHub(num_latents=64, embed_dim=self.d_model, dropout=dropout)
        self.pep_hub = StarFusionHub(num_latents=32, embed_dim=self.d_model, dropout=dropout)

        self.fusion_stack = nn.ModuleList([
            Hla2pepAttentionFusion(**cfg.model.fusion)
        ])
        self.predictor = SequencePredictor(input_size=self.d_model * 2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 引用你的 weight_initial 函数
        weight_initial(self)

    def forward(self, hla_list, pep_list, tcr=None, return_features=False, view_mask=None):
        batch_size = hla_list[0].shape[0]

        # 1. 投影对齐 (List[Tensor] -> List[Tensor])
        # 依然需要 ViewProjector 把维度统一到 64 (d_model)
        proj_hla = self.view_projector(hla_list)
        proj_pep = self.view_projector(pep_list)

        # 2. 独立编码 (不再是 Batch Folding，而是循环/并行编码)
        encoded_hla_views = []
        encoded_pep_views = []

        # 魏征提示：这里虽然用了 Python 循环，但因为模型较大，瓶颈在 compute 而不在 launch
        # 如果追求极致，可以用 torch.vmap (但 Lyra 内部可能有复杂操作不支持 vmap)
        # 所以老老实实循环是最稳的
        for i in range(self.num_views):
            # HLA 编码
            # proj_hla[i]: [B, D, L] (假设 ViewProjector 输出的是 D 在中间)
            # 确认 Lyra 输入需求，如果是 [B, D, L] 则直接传
            _, h_enc = self.hla_encoders[i](proj_hla[i], return_embeddings=True)
            # h_enc: [B, D, L] -> 转置为 [B, L, D] 给 Hub
            encoded_hla_views.append(h_enc.transpose(1, 2))

            # PEP 编码
            _, p_enc = self.pep_encoders[i](proj_pep[i], return_embeddings=True)
            encoded_pep_views.append(p_enc.transpose(1, 2))

        # 3. 门控聚合
        # 此时 encoded_hla_views 是 List[Tensor(B, L, D)]，直接喂给 Hub
        # Hub 内部会自动 stack 并处理 Mask
        hla_latents = self.hla_hub(encoded_hla_views, view_mask=view_mask)
        pep_latents = self.pep_hub(encoded_pep_views, view_mask=view_mask)

        # 4. 跨模态融合
        for fusion in self.fusion_stack:
            out = fusion(hla_latents, pep_latents)

        # 5. 预测
        out_mean = torch.mean(out, dim=-1)
        out_max, _ = torch.max(out, dim=-1)
        final_feat = torch.cat([out_mean, out_max], dim=-1)

        return self.predictor(final_feat)