import torch
import torch.nn as nn


class ContextAwareGatedFusion(nn.Module):
    """
    Context-Aware Gated Fusion (CAGF)
    --------------------------------
    创新点：不仅仅利用 Cross-Attention 捕捉局部交互，还引入 Global Context
    来模拟分子结合时的整体环境（如疏水性环境、空间构象稳定性），
    并通过门控机制动态调节原始特征与交互特征的融合比例。
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__()

        # 1. 标准 Cross-Attention
        # query=Peptide, key=HLA, value=HLA
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # 2. 全局上下文提取器 (Global Context Extractor)
        # 将序列维度压缩为 1，提取整体特征
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 3. 上下文感知门控网络 (Context-Aware Gating Network)
        # 输入维度: d_model * 3 (原始Pep + 交互Attn + 全局Context)
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid()
        )

        # 4. 后处理模块 (Norm + FFN)
        # 采用 Post-Norm 结构，与标准 Transformer Block 保持一致
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),  # 膨胀系数通常为 2 或 4
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, hla_out, pep_out):
        """
        Args:
            hla_out: [Batch, L_hla, D] (Key/Value)
            pep_out: [Batch, L_pep, D] (Query)
        Returns:
            out: [Batch, L_pep, D]
        """
        # B: Batch size, L: Sequence Length, D: d_model

        # === 1. Cross Attention: Peptide 探查 HLA ===
        # attn_out: [B, L_pep, D]
        attn_out, _ = self.cross_attn(query=pep_out, key=hla_out, value=hla_out)

        # === 2. 提取全局上下文 (Global Context) ===
        # Linear 层期望 [B, L, D]，但 Pooling 层期望 [B, D, L]
        # 所以需要 permute
        # [B, L_pep, D] -> [B, D, L_pep] -> Pool -> [B, D, 1]
        global_ctx = self.global_pool(pep_out.permute(0, 2, 1))

        # 还原维度并广播: [B, D, 1] -> [B, 1, D] -> [B, L_pep, D]
        global_ctx = global_ctx.permute(0, 2, 1)
        global_ctx_expanded = global_ctx.expand(-1, pep_out.size(1), -1)

        # === 3. 计算上下文门控 (Gating) ===
        # 拼接三个维度的信息
        # concat_feat: [B, L_pep, 3D]
        concat_feat = torch.cat([pep_out, attn_out, global_ctx_expanded], dim=-1)

        # 计算门控分数 (0~1)
        gate_score = self.gate_net(concat_feat)  # [B, L_pep, D]

        # === 4. 门控融合 (Fusion) ===
        # 融合逻辑：Gate * 交互信息 + (1-Gate) * 原始信息
        fused = gate_score * attn_out + (1 - gate_score) * pep_out

        # 残差连接 + 归一化 (Sublayer Connection)
        # 这里 pep_out 已经在 fused 公式里体现了残差的思想，但加上 norm 更稳健
        out1 = self.norm1(fused + pep_out)

        # === 5. Feed-Forward Network ===
        out2 = self.ffn(out1)
        out = self.norm2(out1 + out2)

        return out