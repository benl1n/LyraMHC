import torch
import torch.nn as nn


class StarFusionHub(nn.Module):
    def __init__(self, num_latents, embed_dim, num_views=4, num_heads=8, dropout=0.1):
        super().__init__()
        # 魏征注：Top-K 参数已被移除，逻辑下放给 mask
        self.num_views = num_views
        self.embed_dim = embed_dim

        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        # [4, 64]
        self.view_embeddings = nn.Parameter(torch.randn(num_views, embed_dim))
        nn.init.normal_(self.view_embeddings, std=embed_dim ** -0.5)

        # batch_first=True 是必须的
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, list_of_features, view_mask=None):
        """
        list_of_features: Tuple/List of [B, D, L] or Tensor [B, V, L, D]
        view_mask: [B, V] (1=Keep, 0=Drop)
        """
        # 1. 堆叠与维度调整
        if isinstance(list_of_features, (list, tuple)):
            # [B, V, D, L]
            features = torch.stack(list_of_features, dim=1)
            # 转置为 [B, V, L, D] 适配 Transformer
            features = features.transpose(-1, -2)
        else:
            features = list_of_features

        B, V, L, D = features.shape

        # 2. 注入视图嵌入 (Broadcasting)
        # [B, V, L, D] + [1, V, 1, D]
        features = features + self.view_embeddings.view(1, V, 1, D)

        # 3. 展平为序列 [B, V*L, D]
        combined_kv = features.flatten(1, 2)

        # 4. 构建 Key Padding Mask (核心逻辑)
        # nn.MultiheadAttention 要求 mask 形状为 [B, Src_Len]
        # True 表示 "IGNORE" (被屏蔽)，False 表示 "KEEP"
        key_padding_mask = None
        if view_mask is not None:
            # view_mask: [B, V] (1=Keep, 0=Drop)

            # Step A: 扩展 mask 到每个氨基酸位点
            # [B, V] -> [B, V, 1] -> [B, V, L]
            expanded_mask = view_mask.unsqueeze(-1).expand(-1, -1, L)

            # Step B: 展平以匹配 combined_kv
            # [B, V*L]
            flat_mask = expanded_mask.reshape(B, -1)

            # Step C: 转换为 Bool (PyTorch 约定: True = Ignore/Masked)
            # 如果 view_mask=0 (Drop), 则 key_padding_mask=True (Ignore)
            key_padding_mask = (flat_mask == 0)

        # 5. Cross-Attention
        latents_expanded = self.latents.unsqueeze(0).expand(B, -1, -1)

        attn_out, _ = self.cross_attn(
            query=latents_expanded,
            key=combined_kv,
            value=combined_kv,
            key_padding_mask=key_padding_mask  # 传入掩码
        )
        latents = self.norm1(latents_expanded + attn_out)

        # 6. Self-Attention & FFN
        latents_out, _ = self.self_attn(latents, latents, latents)
        latents = self.norm2(latents + latents_out)
        latents = self.norm3(latents + self.ffn(latents))

        return latents