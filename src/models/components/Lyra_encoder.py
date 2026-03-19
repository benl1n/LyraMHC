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
        xv = self.in_norm(self.in_proj(u))
        x, v = xv.chunk(2, dim=-1)
        x_conv = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        gate = v * x_conv
        x = self.norm(self.out_proj(gate))
        return x

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability has to be in [0,1), got {p}")
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def forward(self, X):
        if not self.training or self.p == 0.0:
            return X

        if not self.transposed:
            X = rearrange(X, 'b ... d -> b d ...')

        if self.tie:
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2)
        else:
            mask_shape = X.shape

        mask = (torch.rand(*mask_shape, device=X.device) < (1.0 - self.p)).to(X.dtype)

        X = X * mask * (1.0 / (1.0 - self.p))

        if not self.transposed:
            X = rearrange(X, 'b d ... -> b ... d')

        return X


class S4DKernel(nn.Module):
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model

        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.register("log_dt", log_dt, lr)
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        self.register("log_A_real", log_A_real, lr)

        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):

        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)

        C = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:

            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr

            setattr(getattr(self, name), "_optim", optim)


# -------- S4D 本体 --------
class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        self.activation = nn.GELU()

        self.dropout = DropoutNd(dropout, transposed=True) if dropout > 0.0 else nn.Identity()

        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=1),
        )

    def forward(self, u, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)

        B, H, L = u.shape

        k = self.kernel(L=L)  # -> [H, L]

        n = 2 * L
        k_f = torch.fft.rfft(k, n=n)
        u_f = torch.fft.rfft(u, n=n)

        y_f = u_f * k_f

        y = torch.fft.irfft(y_f, n=n)[..., :L]

        y = y + u * self.D.unsqueeze(-1)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.output_linear(y)

        if not self.transposed:
            y = y.transpose(-1, -2)

        return y


class Lyra(nn.Module):
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
            x = norm(x.transpose(-1, -2)).transpose(-1, -2)


        x = x.transpose(-1, -2)

        embeddings = x
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.decoder(x)

        if return_embeddings:
            return x, embeddings
        else:
            return x
