import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.components.Lyra_encoder import Lyra
from src.models.components.fusion_learnable import LiteBiCrossAttentionFusion_l
from src.models.components.Affinity_fusion import AffinityGuidedFusion
from src.models.components.ContextAwareGatedFusion import ContextAwareGatedFusion
from src.models.components.predictors import SequencePredictor


def weight_initial(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            # Kaiming Init 适合 SiLU/ReLU 激活函数，能保持方差一致
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            # 缩放因子(weight)设为1，偏移(bias)设为0
            if m.weight is not None:
                nn.init.constant_(m.weight.data, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
        if "S4D" in str(type(m)) or "Kernel" in str(type(m)):
            continue





class SequenceEncoder(nn.Module):
    def __init__(self, d_input, d_model, dropout, encoder_cfg):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(d_input, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.encoder_fwd = Lyra(**encoder_cfg)
        self.encoder_bwd = Lyra(**encoder_cfg)
        self.output_proj = nn.Linear(d_model * 2, d_model)

        self.dropout_layer = nn.Dropout(dropout)


    def forward(self, x):
        # x: (B, C, L)
        x = self.proj(x)
        x = x.transpose(-1, -2)  # -> (B, L, d_model)
        _, out_fwd = self.encoder_fwd (x, return_embeddings=True)
        x_rev = torch.flip(x, dims=[1])
        _, out_bwd = self.encoder_bwd(x_rev, return_embeddings=True)
        out_bwd = torch.flip(out_bwd, dims=[1])
        out_cat = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.output_proj(out_cat)
        out = self.dropout_layer(out)
        return out



class BiLyraMHC(nn.Module):
    def __init__(self, cfg: DictConfig):

        super(BiLyraMHC, self).__init__()

        self.cfg = cfg

        d_input = cfg.model.params.d_input
        d_model = cfg.model.params.d_model
        dropout = cfg.model.params.dropout
        fusion_cfg = cfg.model.fusion
        predictor_cfg = cfg.model.predictor


        self.hla_encoder = SequenceEncoder(d_input, d_model, dropout, cfg.model.encoder_hla)
        self.pep_encoder = SequenceEncoder(d_input, d_model, dropout, cfg.model.encoder_pep)

        if cfg.task.name == "TCR":
            self.tcr_encoder = SequenceEncoder(d_input, d_model, dropout, cfg.model.encoder_tcr)
            self.fusion_stack_tcr = nn.ModuleList([
                LiteBiCrossAttentionFusion_l(**fusion_cfg)
            ])

        # === Fusion 模块 === 改注意力参数
        self.fusion_stack = nn.ModuleList([
            LiteBiCrossAttentionFusion_l(**fusion_cfg)
        ])

        final_dim = d_model * 2
        pred_input_size = predictor_cfg.get('input_size', final_dim)
        self.predictor = SequencePredictor(input_size=pred_input_size)

    def forward(self,
                hla_a_seqs: torch.Tensor,
                peptides: torch.Tensor,
                tcr: torch.Tensor = None):

        cfg = self.cfg

        # === Encoder ===
        hla_out = self.hla_encoder(hla_a_seqs)
        pep_out = self.pep_encoder(peptides)


        # === 融合 ===
        for fusion in self.fusion_stack:
            out = fusion(hla_out, pep_out)  # 传入两个输入

        if cfg.task.name == "TCR":
            tcr_out = self.tcr_encoder(tcr)
            out = fusion(tcr_out, out)


        out_mean = torch.mean(out,dim=1)
        out_max, _ = torch.max(out,dim=1)
        all_out=torch.cat([out_mean,out_max], dim=-1)
        # === 预测 ===
        ic50 = self.predictor(all_out)
        return ic50



    def load_pretrained_weights(self, ckpt_path, freeze=True):
        print(f"加载预训练权重: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 如果是DataParallel保存的模型，去掉'module.'前缀
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        keep_keys = [
            "hla_encoder",
            "pep_encoder",
            "fusion_stack_pMHC"
        ]

        # 只保留这些模块的参数
        filtered_state = {
            k: v for k, v in state_dict.items()
            if any(k.startswith(prefix) for prefix in keep_keys)
        }

        missing, unexpected = self.load_state_dict(filtered_state, strict=False)
        print(f"已加载 {len(filtered_state)} 个参数.")
        if missing:
            print(f"未加载参数: {missing[:5]} ...")
        if unexpected:
            print(f"预训练模型中多余参数: {unexpected[:5]} ...")