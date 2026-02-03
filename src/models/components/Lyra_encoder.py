import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

class PGC(nn.Module):
    def __init__(self, d_model, expansion_factor=1.0, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.in_proj = nn.Linear(d_model, int(d_model * expansion_factor * 2))
        self.norm = nn.RMSNorm(int(d_model * expansion_factor))
        self.in_norm = nn.RMSNorm(d_model * expansion_factor * 2)
        self.out_proj = nn.Linear(int(d_model * expansion_factor), d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u):
        # self.in_proj(u)
        # u: [batch, seq_len, d_model]
        # -> [batch, seq_len, int(d_model * expansion_factor * 2)]
        # -> RMSNorm进行均方根归一化: [batch, seq_len, int(d_model * expansion_factor * 2)]
        xv = self.in_norm(self.in_proj(u))

        # 将xv在最终维度上均分成两份
        # x: [batch, seq_len, int(d_model * expansion_factor)]
        # v: [batch, seq_len, int(d_model * expansion_factor)]
        x, v = xv.chunk(2, dim=-1)

        # x: [batch, seq_len, int(d_model * expansion_factor)]
        # -> 转置为 [batch, int(d_model * expansion_factor), seq_len]
        # -> 逐通道卷积 (Depthwise Conv1d)
        # -> [batch, int(d_model * expansion_factor), seq_len]
        # -> 转置回来 [batch, seq_len, int(d_model * expansion_factor)]
        x_conv = self.conv(x.transpose(-1, -2)).transpose(-1, -2)

        # v和卷积结果逐元素相乘 (门控机制)
        # gate: [batch, seq_len, int(d_model * expansion_factor)]
        gate = v * x_conv

        # gate: [batch, seq_len, int(d_model * expansion_factor)]
        # -> 全连接投影回原始维度: x: [batch, seq_len, d_model]
        # -> RMSNorm: [batch, seq_len, d_model]
        x = self.norm(self.out_proj(gate))

        return x


# -------- DropoutNd --------
class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability has to be in [0,1), got {p}")
        self.p = p  # scalar
        self.tie = tie  # whether to tie mask across spatial dimensions
        self.transposed = transposed  # True -> data assumed [B, C, L]; False -> [B, L, C]

    def forward(self, X):
        # X can be either:
        #   transposed=True:  X.shape == [B, C, L, ...]  (channels-first)
        #   transposed=False: X.shape == [B, ..., L, C]  (channels-last; we will permute)
        if not self.training or self.p == 0.0:
            return X

        if not self.transposed:
            # convert [B, ..., d] -> [B, d, ...] so channel dim becomes second dim
            # example: [B, L, C] -> [B, C, L]
            X = rearrange(X, 'b ... d -> b d ...')

        # build mask shape:
        # if tie: share same mask across all trailing spatial positions:
        #   X.shape = [B, C, S1, S2, ...]  -> mask_shape = (B, C, 1, 1, ...)
        # else: mask_shape = full X.shape
        if self.tie:
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2)
        else:
            mask_shape = X.shape

        # sample mask in {0,1}, shape = mask_shape
        mask = (torch.rand(*mask_shape, device=X.device) < (1.0 - self.p)).to(X.dtype)

        # apply mask and scale to keep expectation constant
        # X: [B, C, ...], mask broadcast to X -> same shape as X after broadcast
        X = X * mask * (1.0 / (1.0 - self.p))

        if not self.transposed:
            # convert back: [B, C, ...] -> [B, ... , C]
            X = rearrange(X, 'b d ... -> b ... d')

        return X


# -------- S4DKernel (生成卷积核 K) --------
class S4DKernel(nn.Module):
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model  # H = number of channels (d_model)


        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        # store as buffer/parameter via custom register below
        self.register("log_dt", log_dt, lr)

        # C: complex coefficients, shape [H, N//2] (complex)
        # We store C as real view (torch.view_as_real) so parameter is real tensor of shape [H, N//2, 2]
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))  # stored shape: [H, N//2, 2] (real, imag as last dim)

        # log_A_real: real part amplitude log, shape [H, N//2]
        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        self.register("log_A_real", log_A_real, lr)

        # A_imag: imaginary part frequencies, shape [H, N//2]
        # repeat arange to shape [H, N//2], multiply by pi
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):

        dt = torch.exp(self.log_dt)  # shape [H]

        C = torch.view_as_complex(self.C)  # shape [H, N//2]

        A = -torch.exp(self.log_A_real) + 1j * self.A_imag

        dtA = A * dt.unsqueeze(-1)  # shape [H, N//2] (complex)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)

        C = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K  # [H, L]

    def register(self, name, tensor, lr=None):

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            # store as parameter (trainable)
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            # 把优化信息保存在该参数对象上（非 PyTorch 常见做法但可行）
            setattr(getattr(self, name), "_optim", optim)


# -------- S4D 本体 --------
class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        # h = number of channels (d_model)
        self.h = d_model  # H
        self.n = d_state  # N (state dim used inside kernel)
        self.d_output = self.h
        self.transposed = transposed  # whether input is [B, H, L] (True) or [B, L, H] (False)

        # D: residual per-channel scaling, shape [H]
        self.D = nn.Parameter(torch.randn(self.h))

        # kernel: S4DKernel producing per-channel kernel K of shape [H, L]
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        self.activation = nn.GELU()

        # dropout: custom DropoutNd that expects channels-first tensors
        self.dropout = DropoutNd(dropout, transposed=True) if dropout > 0.0 else nn.Identity()

        # output linear: 1x1 Conv to expand channels  -> GLU to gate -> back to H
        # input to Conv1d expected shape: [B, C_in, L] = [B, H, L]
        # Conv1d(H, 2H, kernel_size=1) -> output [B, 2H, L]
        # GLU(dim=1) splits channel dim into 2 and gates -> [B, H, L]
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=1),
        )

    def forward(self, u, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)

        B, H, L = u.shape

        # 1) compute kernel K of length L: K shape [H, L]
        k = self.kernel(L=L)  # -> [H, L]

        # 2) FFT-based convolution:
        #    we use n = 2*L to avoid circular wrap-around and compute via rfft/irfft
        n = 2 * L
        # rfft along last dim:
        # k_f: [H, n//2+1]  (complex-valued in rfft output representation)
        k_f = torch.fft.rfft(k, n=n)  # k: [H, L] -> rfft -> [H, n//2+1]
        # u_f: [B, H, n//2+1]
        u_f = torch.fft.rfft(u, n=n)  # u: [B, H, L] -> rfft -> [B, H, n//2+1]

        # pointwise multiply in freq domain, broadcasting k_f across batch:
        # u_f * k_f -> [B, H, n//2+1]
        y_f = u_f * k_f

        # inverse rfft -> real time-domain signal length n, then take first L positions
        # irfft(..., n=n) -> [B, H, n], we take [..., :L] -> [B, H, L]
        y = torch.fft.irfft(y_f, n=n)[..., :L]  # -> [B, H, L]

        # 3) add diagonal residual D per channel: D.unsqueeze(-1) -> [H,1], broadcast to [B,H,L]
        y = y + u * self.D.unsqueeze(-1)  # [B,H,L]

        # 4) nonlinearity + dropout (still channels-first)
        y = self.activation(y)  # [B,H,L]
        y = self.dropout(y)  # [B,H,L]

        # 5) output projection: Conv1d expects [B, C, L]
        y = self.output_linear(y)  # Conv1d -> [B,2H,L], GLU -> [B,H,L]

        # 6) if input was channels-last, convert back to [B, L, H]
        if not self.transposed:
            y = y.transpose(-1, -2)  # -> [B, L, H]

        # final shape: [B, H, L] (if transposed=True) or [B, L, H] otherwise
        return y


class Lyra(nn.Module):
    """
    Lyra model incorporating PGC and S4D layers for sequence processing.

    Parameters:
    - model_dimension: Internal dimension of S4D layers and projection layers before and after PGC.
    - pgc_configs (list of tuples): Configuration for PGC layers, where each tuple contains
    (hidden dimension, number of layers in the PGC module).
    - num_s4 (int): Number of S4D layers.
    - d_input (int): Dimensionality of the input features.
    - d_output (int, optional): Dimensionality of the output features. Defaults to 10.
    - dropout (float, optional): Dropout rate for regularization. Defaults to 0.2.
    - prenorm (bool, optional): Whether to use pre-normalization. Defaults to True.
    """

    def __init__(
            self,
            model_dimension,
            pgc_configs,
            num_s4,
            d_input,
            d_output=10,
            dropout=0.2,
            prenorm=True,
            final_dropout=0.2
    ):
        super().__init__()

        self.encoder = nn.Linear(d_input, model_dimension)
        self.pgc_layers = nn.ModuleList()
        for config in pgc_configs:
            pgc_hidden_dimension, num_layers = config
            for _ in range(num_layers):
                self.pgc_layers.append(PGC(model_dimension, pgc_hidden_dimension,
                                           dropout))
            self.prenorm = prenorm

            # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_s4):
            self.s4_layers.append(
                S4D(model_dimension, dropout=dropout, transposed=True)
            )
            self.norms.append(nn.RMSNorm(model_dimension))
            self.dropouts.append(nn.Dropout(dropout))
            self.decoder = nn.Linear(model_dimension, d_output)
            self.dropout = nn.Dropout(final_dropout)


    def forward(self, x, return_embeddings=False):
        x = self.encoder(x)
        for pgc_layer in self.pgc_layers:
            x = pgc_layer(x)
        x = x.transpose(-1, -2)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z = layer(z)
            z = dropout(z)
            x = z + x
        if not self.prenorm:
            # Postnorm
            x = norm(x.transpose(-1, -2)).transpose(-1, -2)


        x = x.transpose(-1, -2)

        embeddings = x
        x = x.mean(dim=1)
        x = self.dropout(x)  # (B, d_model) -> (B, d_output)
        x = self.decoder(x)

        if return_embeddings:
            return x, embeddings
        else:
            return x
