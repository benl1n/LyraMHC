import torch
from torch import nn


class AffinityGuidedFusion(nn.Module):
    def __init__(self, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # === 创新点：亲和力探测器 (Affinity Probe) ===
        # 专门用来计算 "肽段残基 i" 和 "MHC 残基 j" 是否匹配的轻量级网络
        self.affinity_proj = nn.Linear(d_model, d_model // 2)

        # 上下文门控 (方案A的简化版)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout)
        )

    def forward(self, hla_out, pep_out):
        B, L_pep, D = pep_out.shape
        _, L_hla, _ = hla_out.shape

        # 1. 计算显式的亲和力矩阵 (Explicit Affinity Matrix)
        # 将维度降下来计算，模拟特异性结合
        pep_low = self.affinity_proj(pep_out)  # [B, L_pep, D/2]
        hla_low = self.affinity_proj(hla_out)  # [B, L_hla, D/2]

        # 计算匹配分数 [B, L_pep, L_hla]
        # 这里模拟的是：这个肽段残基能不能塞进这个 HLA 口袋
        affinity_score = torch.matmul(pep_low, hla_low.transpose(-2, -1))

        # 2. 将亲和力矩阵作为 Attention Mask 注入
        # 这里的逻辑是：如果亲和力低，Attention 就不应该关注那里
        # 我们把 affinity_score 缩放后加到 attention mask 里 (需要 MHA 支持 attn_mask 参数)

        # 但标准 MHA 的 attn_mask 通常是 bool 或 additive。
        # 我们这里用一个 Trick：把 Affinity Score 当作 bias 加到 Query 上？
        # 不，最稳健的方法是：用 Affinity Score 对 Value 进行预加权

        #

        # 这里为了稳健，我们采用 "Soft-Masking" 策略
        # 我们让 MHA 自己去学，但是我们把 Affinity Score 作为一个额外的特征拼接到 Query 里
        # 让 Query 知道 "我也许应该关注 HLA 的第 j 个位置"

        # 简化版创新：Affinity-Enhanced Query
        # 利用 Max Pooling 找出每个肽段残基最匹配的 HLA 信号强度
        max_affinity, _ = affinity_score.max(dim=-1, keepdim=True)  # [B, L_pep, 1]

        # 将这个 "最大匹配强度" 注入到 Peptide Query 中
        # 告诉模型：这个氨基酸在 HLA 上找到了很强的伙伴，你要重点关注！
        pep_enhanced = pep_out + max_affinity  # 简单的广播相加，或者过个 Linear

        # 3. 标准 Cross Attention (但在 Query 里夹带了私货)
        attn_out, _ = self.cross_attn(query=pep_enhanced, key=hla_out, value=hla_out)

        # 4. 门控融合
        concat = torch.cat([pep_out, attn_out], dim=-1)
        gate = self.gate(concat)
        fused = gate * attn_out + (1 - gate) * pep_out

        return self.norm(fused + self.ffn(fused))