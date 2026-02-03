import torch
from torch import nn


class LiteBiCrossAttentionFusion(nn.Module):
    def __init__(self, d_model:int = 64,
                 hidden_dim: int = 64,
                 n_heads: int = 4,
                 dropout: int = 0.2,
                 max_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.max_len = max_len

        self.proj_hla = nn.Linear(d_model, hidden_dim)
        self.proj_pep = nn.Linear(d_model, hidden_dim)

        self.rel_bias = nn.Parameter(torch.zeros((n_heads, max_len, max_len)))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

        self.norm_hla = nn.LayerNorm(hidden_dim)
        self.norm_pep = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

        self.gate_hla = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.gate_pep = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, hla_out, pep_out):

        B, L_pep, _ = pep_out.shape
        _, L_hla, _ = hla_out.shape

        hla = self.proj_hla(hla_out)
        pep = self.proj_pep(pep_out)

        q_pep = pep.view(B, L_pep, self.n_heads, -1).transpose(1, 2)
        k_hla = hla.view(B, L_hla, self.n_heads, -1).transpose(1, 2)
        v_hla = hla.view(B, L_hla, self.n_heads, -1).transpose(1, 2)


        attn_score = torch.matmul(q_pep, k_hla.transpose(-2, -1)) / (q_pep.size(-1) ** 0.5)
        attn_score = attn_score + self.rel_bias[:, :L_pep, :L_hla]
        attn_weight = attn_score.softmax(dim=-1)
        pep2hla = torch.matmul(attn_weight, v_hla).transpose(1, 2).contiguous().view(B, L_pep, -1)

        q_hla = hla.view(B, L_hla, self.n_heads, -1).transpose(1, 2)
        k_pep = pep.view(B, L_pep, self.n_heads, -1).transpose(1, 2)
        v_pep = pep.view(B, L_pep, self.n_heads, -1).transpose(1, 2)

        attn_score2 = torch.matmul(q_hla, k_pep.transpose(-2, -1)) / (q_hla.size(-1) ** 0.5)
        attn_score2 = attn_score2 + self.rel_bias[:, :L_hla, :L_pep]
        attn_weight2 = attn_score2.softmax(dim=-1)
        hla2pep = torch.matmul(attn_weight2, v_pep).transpose(1, 2).contiguous().view(B, L_hla, -1)

        pep2hla = pep + self.ffn(self.norm_pep(pep2hla))
        # return pep2hla
        hla2pep = hla + self.ffn(self.norm_hla(hla2pep))
        # return hla2pep

        fused = self.gate_hla(hla2pep[:, :L_pep, :]) * hla2pep[:, :L_pep, :] \
              + self.gate_pep(pep2hla) * pep2hla \
              + 0.3 * (pep2hla + hla2pep[:, :L_pep, :])

        return self.fc_out(fused)  # [B, L_pep, D]



class Hla2pepAttentionFusion(nn.Module):
    def __init__(self, d_model:int = 64,
                 hidden_dim: int = 64,
                 n_heads: int = 4,
                 dropout: int = 0.2,
                 max_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.max_len = max_len

        self.proj_hla = nn.Linear(d_model, hidden_dim)
        self.proj_pep = nn.Linear(d_model, hidden_dim)

        self.rel_bias = nn.Parameter(torch.zeros((n_heads, max_len, max_len)))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

        self.norm_hla = nn.LayerNorm(hidden_dim)
        self.norm_pep = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

        self.gate_hla = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.gate_pep = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, hla_out, pep_out):

        B, L_pep, _ = pep_out.shape
        _, L_hla, _ = hla_out.shape

        hla = self.proj_hla(hla_out)
        pep = self.proj_pep(pep_out)

        q_hla = hla.view(B, L_hla, self.n_heads, -1).transpose(1, 2)
        k_pep = pep.view(B, L_pep, self.n_heads, -1).transpose(1, 2)
        v_pep = pep.view(B, L_pep, self.n_heads, -1).transpose(1, 2)

        attn_score2 = torch.matmul(q_hla, k_pep.transpose(-2, -1)) / (q_hla.size(-1) ** 0.5)
        attn_score2 = attn_score2 + self.rel_bias[:, :L_hla, :L_pep]
        attn_weight2 = attn_score2.softmax(dim=-1)
        hla2pep = torch.matmul(attn_weight2, v_pep).transpose(1, 2).contiguous().view(B, L_hla, -1)


        hla2pep = hla + self.ffn(self.norm_hla(hla2pep))

        return self.fc_out(hla2pep)  # [B, L_pep, D]

