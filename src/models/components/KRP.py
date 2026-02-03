import torch
from torch import nn



class KernelResidualProjector(nn.Module):
    def __init__(self, d_model, num_views=4, kernel_type='mlp'):
        super().__init__()
        self.num_residual_views = num_views - 1
        self.d_model = d_model

        # 魏征修正：引入非线性核
        if kernel_type == 'linear':
            # 旧版：线性核
            self.kernel_net = nn.Linear(d_model, self.num_residual_views * d_model)
        elif kernel_type == 'mlp':
            # 新版：MLP 核 (Universal Approximator)
            # 升维 -> 激活 -> 降维/投影
            hidden_dim = d_model * 2  # 中间层升维，捕捉高阶特征
            self.kernel_net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),  # 或者 SiLU，比 ReLU 更适合生物特征
                nn.Linear(hidden_dim, self.num_residual_views * d_model)  # 投影回目标空间
            )

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_views)])

    def forward(self, projected_views):
        # ... (forward 逻辑与之前完全一致，只是 self.merged_mapping 变成了 self.kernel_net)
        # 输入: [B, 4, L, D] (假设你已经用了优化后的 ViewProjector)

        # 1. 拆分基准和其他视图
        base_view = projected_views[0]  # [1024, 34, 64]

        # --- Step 2: 诸侯结盟 (Stacking Others) ---
        # 魏征提示：不要在循环里一个个减！先把后3个视图拼起来，变成一大块肉。
        # list[Tensor] -> Tensor [B, 3, L, D] (1024, 3, 34, 64)
        other_views = torch.stack(projected_views[1:], dim=1)

        # 2. 核空间映射 (Identity -> Expected Properties)
        # [B, L, D] -> [B, L, 3*D]
        all_projections = self.kernel_net(base_view)

        # --- Step 4: 维度重塑与对齐 ---
        # 目标：匹配 other_views 的形状 [B, 3, L, D]
        # [B, L, 3*D] -> [B, L, 3, D] -> [B, 3, L, D]
        all_projections = all_projections.view(
            base_view.shape[0], base_view.shape[1], self.num_residual_views, self.d_model
        ).transpose(1, 2)

        # --- Step 5: 并行残差计算 (核心加速点) ---
        # [B, 3, L, D] - [B, 3, L, D] -> 一次指令完成数十万次减法
        residuals = other_views - all_projections

        # --- Step 6: 归一化与还原 ---
        # 由于每个视图有独立的 LayerNorm 参数 (self.norms[i])，这里必须循环
        # 除非你改用 GroupNorm，否则这是唯一的串行点
        final_views = [self.norms[0](base_view)]

        for i in range(self.num_residual_views):
            # residuals[:, i] 取出来是 [B, L, D]
            normed_res = self.norms[i + 1](residuals[:, i, :, :])
            final_views.append(normed_res)

        # 返回 List 以适配下游 (如果下游也改成了 Tensor 输入，这里可以用 torch.stack 返回)
        return final_views