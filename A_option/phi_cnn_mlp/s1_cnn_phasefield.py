"""
scheme1_cnn_phasefield.py
方案1: CNN学习φ + MLP学习V (Phase-Field Energy)
美式看跌期权自由边界求解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from scipy.interpolate import RegularGridInterpolator

# ============================================================
# 全局配置
# ============================================================
class Config:
    # 期权参数
    S_max = 200.0
    T = 1.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    
    # Phase-Field 参数
    eps = 0.05            # 界面宽度
    lambda_int = 0.01     # 界面能权重 (关键: 不能太大)
    lambda_ex = 3.0       # 行权约束权重
    lambda_pde = 1.0      # PDE残差权重
    lambda_anchor = 20.0     # 行权区锚定权重
    lambda_phi_balance = 2.0 # φ 体积平衡约束
    lambda_bc = 50.0      # 边界条件权重
    lambda_data = 0.0     # Stage 2 中可选的 FDM 监督 (设为0表示纯能量驱动)
    
    # 网格
    grid_S = 128          # φ 网格 S 方向点数
    grid_t = 64           # φ 网格 t 方向点数
    
    # MLP (SIREN) 参数
    mlp_hidden = 256
    mlp_layers = 4
    omega_0 = 5.0         # 修正: 降低 ω₀
    
    # CNN (U-Net) 参数
    cnn_base_ch = 32
    cnn_levels = 3
    cnn_in_channels = 3   # 修正: (S_norm, t_norm, V_pred)
    
    # FDM 参数
    fdm_S = 500
    fdm_t = 2000
    
    
    # 训练参数
    stage1_epochs = 3000       # MLP 预热
    stage1_lr = 2e-3
    stage15_epochs = 800       # 新增: φ warm-start
    stage15_lr = 1e-3
    stage2_epochs = 20000      # 联合训练
    stage2_lr_mlp = 5e-4
    stage2_lr_cnn = 1e-2
    stage3_steps = 50          # L-BFGS 精调
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42

cfg = Config()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)


# ============================================================
# FDM 求解器 (Crank-Nicolson, 美式看跌)
# ============================================================
def generate_fdm_american_put(cfg):
    """Crank-Nicolson 隐式求解美式看跌期权"""
    S_max, T, K, r, sigma = cfg.S_max, cfg.T, cfg.K, cfg.r, cfg.sigma
    Ns, Nt = cfg.fdm_S, cfg.fdm_t
    
    dS = S_max / Ns
    dt = T / Nt
    
    S = np.linspace(0, S_max, Ns + 1)
    V = np.maximum(K - S, 0.0)  # 终端条件
    
    # 构建三对角矩阵系数
    j = np.arange(1, Ns)
    aj = 0.5 * dt * (sigma**2 * j**2 - r * j)
    bj = 1 + dt * (sigma**2 * j**2 + r)
    cj = 0.5 * dt * (-sigma**2 * j**2 - r * j) * (-1)  # 注意符号
    
    # 实际上用标准 CN 公式:
    # 隐式部分系数
    alpha = 0.25 * dt * (sigma**2 * j**2 - r * j)
    beta_impl = -(0.5 * dt * (sigma**2 * j**2 + r))
    gamma = 0.25 * dt * (sigma**2 * j**2 + r * j)  # 修正符号
    
    # 重新构建 (使用标准 Wilmott 公式)
    j = np.arange(1, Ns)
    sigma2 = sigma**2
    
    # A_impl * V^{n} = A_expl * V^{n+1} (反向时间)
    # 从 t=T 向 t=0 递推
    
    a = 0.25 * dt * (sigma2 * j**2 - r * j)
    b = -0.5 * dt * (sigma2 * j**2 + r)
    c = 0.25 * dt * (sigma2 * j**2 + r * j)
    
    # 隐式矩阵 M1: I - B (对 V^n)
    # 显式矩阵 M2: I + B (对 V^{n+1})
    M = Ns - 1
    M1 = np.zeros((M, M))
    M2 = np.zeros((M, M))
    
    for i in range(M):
        M1[i, i] = 1 - b[i]
        M2[i, i] = 1 + b[i]
        if i > 0:
            M1[i, i-1] = -a[i]
            M2[i, i-1] = a[i]
        if i < M - 1:
            M1[i, i+1] = -c[i]
            M2[i, i+1] = c[i]
    
    # 存储所有时间步的解
    V_all = np.zeros((Nt + 1, Ns + 1))
    V_all[Nt, :] = np.maximum(K - S, 0.0)  # t=T
    
    V_interior = V[1:Ns].copy()
    
    from numpy.linalg import solve
    
    for n in range(Nt - 1, -1, -1):
        # 右端向量
        rhs = M2 @ V_interior
        
        # 边界条件修正
        # V(0, t) = K
        rhs[0] += a[0] * K + (-a[0]) * K  # 隐式和显式边界
        # 简化: 左边界 V(0)=K, 右边界 V(S_max)=0
        rhs[0] += (a[0] + a[0]) * K  # 需要仔细处理
        
        # 实际上让我用更简单直接的方法
        # 直接用 fully implicit 方案 (更稳定)
        pass
    
    # ---- 改用 fully implicit 方案 (简单可靠) ----
    V = np.maximum(K - S, 0.0)
    V_all[Nt, :] = V.copy()
    
    for n in range(Nt - 1, -1, -1):
        # 构建三对角系统 (fully implicit)
        a_coef = np.zeros(Ns + 1)
        b_coef = np.zeros(Ns + 1)
        c_coef = np.zeros(Ns + 1)
        d_coef = np.zeros(Ns + 1)
        
        for jj in range(1, Ns):
            a_coef[jj] = 0.5 * dt * (r * jj - sigma2 * jj**2)
            b_coef[jj] = 1 + dt * (sigma2 * jj**2 + r)
            c_coef[jj] = -0.5 * dt * (r * jj + sigma2 * jj**2)
            d_coef[jj] = V[jj]
        
        # 边界条件
        b_coef[0] = 1.0
        d_coef[0] = K  # V(0, t) = K (put)
        b_coef[Ns] = 1.0
        d_coef[Ns] = 0.0  # V(S_max, t) = 0
        
        # Thomas 算法 (追赶法)
        V_new = thomas_solve(a_coef, b_coef, c_coef, d_coef, Ns + 1)
        
        # 美式约束: V >= max(K-S, 0)
        payoff = np.maximum(K - S, 0.0)
        V_new = np.maximum(V_new, payoff)
        
        V = V_new.copy()
        V_all[n, :] = V.copy()
    
    t_grid = np.linspace(0, T, Nt + 1)
    return S, t_grid, V_all


def thomas_solve(a, b, c, d, n):
    """三对角矩阵的 Thomas 算法"""
    a_ = a.copy()
    b_ = b.copy()
    c_ = c.copy()
    d_ = d.copy()
    
    # 前向消元
    for i in range(1, n):
        if abs(b_[i-1]) < 1e-15:
            continue
        m = a_[i] / b_[i-1]
        b_[i] -= m * c_[i-1]
        d_[i] -= m * d_[i-1]
    
    # 回代
    x = np.zeros(n)
    x[n-1] = d_[n-1] / b_[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (d_[i] - c_[i] * x[i+1]) / b_[i]
    
    return x


# ============================================================
# 准备 FDM 参考解与插值器
# ============================================================
print("生成 FDM 参考解...")
S_fdm, t_fdm, V_fdm = generate_fdm_american_put(cfg)

# 构建插值器: (t, S) -> V
fdm_interp = RegularGridInterpolator(
    (t_fdm, S_fdm), V_fdm, 
    method='linear', bounds_error=False, fill_value=None
)

# 提取 FDM 自由边界 (用于比较)
def extract_fdm_boundary(S_fdm, t_fdm, V_fdm, K):
    """提取自由边界 S*(t): V(S*(t), t) = K - S*(t)"""
    payoff = np.maximum(K - S_fdm, 0.0)
    boundary = np.zeros(len(t_fdm))
    for i in range(len(t_fdm)):
        diff = np.abs(V_fdm[i, :] - payoff)
        # 找到从左到右第一个 V > payoff 的点
        exercised = V_fdm[i, :] <= payoff + 1e-6
        if np.any(~exercised):
            idx = np.where(~exercised)[0][0]
            boundary[i] = S_fdm[max(0, idx - 1)]
        else:
            boundary[i] = S_fdm[-1]
    return boundary

fdm_boundary = extract_fdm_boundary(S_fdm, t_fdm, V_fdm, cfg.K)
print(f"FDM 参考解完成. V_fdm shape: {V_fdm.shape}, "
      f"V range: [{V_fdm.min():.4f}, {V_fdm.max():.4f}]")


# ============================================================
# SIREN 层 与 Value MLP
# ============================================================
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, omega_0=5.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            dim = self.linear.weight.shape[1]
            if self.is_first:
                bound = 1.0 / dim
            else:
                bound = np.sqrt(6.0 / dim) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ValueMLP(nn.Module):
    """SIREN MLP: (S_norm, t_norm) -> V_norm"""
    def __init__(self, hidden_dim=256, num_layers=4, omega_0=5.0):
        super().__init__()
        layers = []
        layers.append(SirenLayer(2, hidden_dim, omega_0=omega_0, is_first=True))
        for _ in range(num_layers - 2):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, 1)
        # 最后一层用小初始化
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_dim) / omega_0
            self.final.weight.uniform_(-bound, bound)
            self.final.bias.zero_()
    
    def forward(self, S_norm, t_norm):
        """
        S_norm: (N,1) in [0,1]
        t_norm: (N,1) in [0,1]
        返回: V_norm (N,1) — 归一化值
        """
        x = torch.cat([S_norm, t_norm], dim=-1)  # (N, 2)
        h = self.net(x)
        return self.final(h)


# ============================================================
# 轻量 U-Net (CNN for φ)
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
    
    def forward(self, x):
        return self.block(x)


class PhiUNet(nn.Module):
    """
    轻量 U-Net: 输入 (B, C_in, H, W) -> 输出 (B, 1, H, W) ∈ (0,1)
    C_in = 3: (S_norm_grid, t_norm_grid, V_pred_grid)
    """
    def __init__(self, in_channels=3, base_ch=32, levels=3):
        super().__init__()
        self.levels = levels
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(levels):
            out_ch = base_ch * (2 ** i)
            self.encoders.append(ConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch
        
        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch * 2)
        ch = ch * 2
        
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(levels - 1, -1, -1):
            out_ch = base_ch * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(ch, out_ch, 2, stride=2))
            self.decoders.append(ConvBlock(out_ch * 2, out_ch))  # concat with skip
            ch = out_ch
        
        # Final 1x1
        self.final = nn.Conv2d(ch, 1, 1)
    
    def forward(self, x):
        # Encoder
        skips = []
        h = x
        for enc, pool in zip(self.encoders, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = upconv(h)
            # 处理尺寸不匹配
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)
        
        return torch.sigmoid(self.final(h))


# ============================================================
# 构建网格与坐标
# ============================================================
def build_grid(cfg):
    """构建 φ 的规则网格坐标 (归一化到 [0,1])"""
    S_vals = torch.linspace(0, 1, cfg.grid_S)  # S_norm
    t_vals = torch.linspace(0, 1, cfg.grid_t)  # t_norm
    
    # (grid_t, grid_S) meshgrid
    tt, ss = torch.meshgrid(t_vals, S_vals, indexing='ij')
    
    # CNN 输入: (1, C, grid_t, grid_S)
    S_grid = ss.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t_grid = tt.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    
    return S_grid.to(cfg.device), t_grid.to(cfg.device), S_vals.to(cfg.device), t_vals.to(cfg.device)

S_grid, t_grid, S_vals, t_vals = build_grid(cfg)

# 用于 MLP 计算的展平坐标 (需要 requires_grad)
S_flat = (S_grid.squeeze() * cfg.S_max).reshape(-1, 1)  # 原始量纲
t_flat = (t_grid.squeeze() * cfg.T).reshape(-1, 1)
N_grid = S_flat.shape[0]


def get_fdm_on_grid(cfg, fdm_interp):
    """在 PINN 网格上插值 FDM 解"""
    S_np = np.linspace(0, cfg.S_max, cfg.grid_S)
    t_np = np.linspace(0, cfg.T, cfg.grid_t)
    tt_np, ss_np = np.meshgrid(t_np, S_np, indexing='ij')
    points = np.stack([tt_np.ravel(), ss_np.ravel()], axis=-1)
    V_ref = fdm_interp(points).reshape(cfg.grid_t, cfg.grid_S)
    return torch.tensor(V_ref, dtype=torch.float32).to(cfg.device)

V_fdm_grid = get_fdm_on_grid(cfg, fdm_interp)  # (grid_t, grid_S)
V_fdm_norm = V_fdm_grid / cfg.K  # 归一化

# Payoff on grid
payoff_grid = torch.clamp(cfg.K - S_grid.squeeze() * cfg.S_max, min=0.0) / cfg.K  # 归一化 (grid_t, grid_S)


# ============================================================
# 初始化网络
# ============================================================
v_net = ValueMLP(
    hidden_dim=cfg.mlp_hidden, 
    num_layers=cfg.mlp_layers,
    omega_0=cfg.omega_0
).to(cfg.device)

phi_net = PhiUNet(
    in_channels=cfg.cnn_in_channels,
    base_ch=cfg.cnn_base_ch,
    levels=cfg.cnn_levels
).to(cfg.device)

print(f"V_net 参数量: {sum(p.numel() for p in v_net.parameters()):,}")
print(f"Phi_net 参数量: {sum(p.numel() for p in phi_net.parameters()):,}")


# ============================================================
# 辅助函数: 计算 BS 残差 (autograd)
# ============================================================
def compute_bs_residual(v_net, S_norm, t_norm, cfg):
    """
    计算 Black-Scholes PDE 残差: ℒV = V_t + 0.5σ²S²V_SS + rSV_S - rV
    输入: S_norm, t_norm ∈ [0,1], requires_grad=True
    V_net 输出: V_norm = V / K
    原始变量: V = K * V_norm, S = S_max * S_norm, t = T * t_norm
    
    链式法则:
        V_t = K/T * V_norm_t_norm
        V_S = K/S_max * V_norm_S_norm
        V_SS = K/S_max² * V_norm_S_norm_S_norm
    """
    S_n = S_norm.clone().requires_grad_(True)
    t_n = t_norm.clone().requires_grad_(True)
    
    V_norm = v_net(S_n, t_n)  # (N, 1)
    
    # 一阶导
    grad_outputs = torch.ones_like(V_norm)
    dV_dS = torch.autograd.grad(V_norm, S_n, grad_outputs=grad_outputs, create_graph=True)[0]
    dV_dt = torch.autograd.grad(V_norm, t_n, grad_outputs=grad_outputs, create_graph=True)[0]
    
    # 二阶导
    d2V_dS2 = torch.autograd.grad(dV_dS, S_n, grad_outputs=torch.ones_like(dV_dS), create_graph=True)[0]
    
    # 转换到原始变量
    S = S_n * cfg.S_max
    V_t = (cfg.K / cfg.T) * dV_dt
    V_S = (cfg.K / cfg.S_max) * dV_dS
    V_SS = (cfg.K / cfg.S_max**2) * d2V_dS2
    V = cfg.K * V_norm
    
    # BS 残差: V_t + 0.5*σ²*S²*V_SS + r*S*V_S - r*V = 0
    residual = V_t + 0.5 * cfg.sigma**2 * S**2 * V_SS + cfg.r * S * V_S - cfg.r * V
    
    # 归一化残差 (除以K使量纲一致)
    residual_norm = residual / cfg.K
    
    return V_norm, residual_norm


# ============================================================
# 辅助函数: φ 的有限差分梯度
# ============================================================
def phi_gradient_fd(phi, dS_norm, dt_norm):
    """
    中心差分计算 |∇φ|²
    phi: (1, 1, H, W) or (H, W)
    """
    if phi.dim() == 2:
        phi = phi.unsqueeze(0).unsqueeze(0)
    
    # S 方向 (W 轴)
    dphi_dS = torch.zeros_like(phi)
    dphi_dS[:, :, :, 1:-1] = (phi[:, :, :, 2:] - phi[:, :, :, :-2]) / (2 * dS_norm)
    dphi_dS[:, :, :, 0] = (phi[:, :, :, 1] - phi[:, :, :, 0]) / dS_norm
    dphi_dS[:, :, :, -1] = (phi[:, :, :, -1] - phi[:, :, :, -2]) / dS_norm
    
    # t 方向 (H 轴)
    dphi_dt = torch.zeros_like(phi)
    dphi_dt[:, :, 1:-1, :] = (phi[:, :, 2:, :] - phi[:, :, :-2, :]) / (2 * dt_norm)
    dphi_dt[:, :, 0, :] = (phi[:, :, 1, :] - phi[:, :, 0, :]) / dt_norm
    dphi_dt[:, :, -1, :] = (phi[:, :, -1, :] - phi[:, :, -2, :]) / dt_norm
    
    grad_sq = dphi_dS**2 + dphi_dt**2
    return grad_sq.squeeze()  # (H, W)


# ============================================================
# Stage 1: MLP 预热 (FDM 监督)
# ============================================================
print("\n" + "=" * 70)
print("Stage 1: MLP预热 (FDM监督)")
print("=" * 70)

optimizer_s1 = torch.optim.AdamW(v_net.parameters(), lr=cfg.stage1_lr, weight_decay=1e-5)
scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=cfg.stage1_epochs, eta_min=1e-5)

history_s1 = {'mse': [], 'bc': [], 'rel_err': []}

for epoch in range(1, cfg.stage1_epochs + 1):
    v_net.train()
    
    # 在网格点上计算 V
    S_in = S_grid.squeeze().reshape(-1, 1).clone() / 1.0  # 已归一化 [0,1]
    t_in = t_grid.squeeze().reshape(-1, 1).clone() / 1.0
    
    V_pred_norm = v_net(S_in.to(cfg.device), t_in.to(cfg.device))  # (N, 1)
    V_target_norm = V_fdm_norm.reshape(-1, 1)  # (N, 1)
    
    # MSE loss (归一化空间)
    mse_loss = F.mse_loss(V_pred_norm, V_target_norm)
    
    # BC loss
    # V(S=0, t) = K -> V_norm(0, t) = 1
    n_bc = cfg.grid_t
    S_bc0 = torch.zeros(n_bc, 1, device=cfg.device)
    t_bc0 = torch.linspace(0, 1, n_bc, device=cfg.device).unsqueeze(1)
    V_bc0 = v_net(S_bc0, t_bc0)
    bc_loss_left = F.mse_loss(V_bc0, torch.ones_like(V_bc0))
    
    # V(S_max, t) = 0 -> V_norm(1, t) = 0
    S_bc1 = torch.ones(n_bc, 1, device=cfg.device)
    t_bc1 = t_bc0.clone()
    V_bc1 = v_net(S_bc1, t_bc1)
    bc_loss_right = F.mse_loss(V_bc1, torch.zeros_like(V_bc1))
    
    # V(S, T) = max(K-S, 0)/K -> terminal
    n_tc = cfg.grid_S
    S_tc = torch.linspace(0, 1, n_tc, device=cfg.device).unsqueeze(1)
    t_tc = torch.ones(n_tc, 1, device=cfg.device)  # t_norm = 1 对应 t = T
    V_tc = v_net(S_tc, t_tc)
    V_tc_target = torch.clamp(1.0 - S_tc * cfg.S_max / cfg.K, min=0.0)
    bc_loss_terminal = F.mse_loss(V_tc, V_tc_target)
    
    bc_loss = bc_loss_left + bc_loss_right + bc_loss_terminal
    
    total_loss = mse_loss + cfg.lambda_bc * bc_loss
    
    optimizer_s1.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
    optimizer_s1.step()
    scheduler_s1.step()
    
    # 记录
    with torch.no_grad():
        V_pred_orig = V_pred_norm.reshape(cfg.grid_t, cfg.grid_S) * cfg.K
        V_target_orig = V_fdm_grid
        rel_err = torch.norm(V_pred_orig - V_target_orig) / (torch.norm(V_target_orig) + 1e-8)
    
    history_s1['mse'].append(mse_loss.item())
    history_s1['bc'].append(bc_loss.item())
    history_s1['rel_err'].append(rel_err.item())
    
    if epoch % 500 == 0:
        with torch.no_grad():
            V_p = V_pred_norm.reshape(-1)
            V_t = V_target_norm.reshape(-1)
        print(f"  Epoch {epoch}/{cfg.stage1_epochs} | "
              f"MSE(norm)={mse_loss.item():.6f} | BC={bc_loss.item():.6f} | "
              f"RelErr={rel_err.item():.4f} | "
              f"V pred [mean={V_p.mean():.4f}, std={V_p.std():.4f}, "
              f"min={V_p.min():.4f}, max={V_p.max():.4f}] | "
              f"V target [mean={V_t.mean():.4f}, std={V_t.std():.4f}]")

print(f"Stage 1 完成. 最终 RelErr = {history_s1['rel_err'][-1]:.6f}")


# ============================================================
# Stage 1.5: φ Warm-Start
# ============================================================
print("\n" + "=" * 70)
print("Stage 1.5: φ Warm-Start (用 FDM 解初始化 CNN)")
print("=" * 70)

# 方法: 比较 V_fdm 与 Payoff, gap > threshold 则继续区(φ=1), 否则行权区(φ=0)
with torch.no_grad():
    # 方法: 在 FDM 的精细网格上判断行权区 (无插值误差)
    payoff_fdm = np.maximum(cfg.K - S_fdm, 0.0)  # (Ns+1,)
    
    # 对每个时间步, 判断 V_fdm[i,:] 是否 "实质上等于" payoff
    # 用相对阈值: |V - payoff| / K < tol
    tol = 1e-4  # FDM 精度量级
    phi_fdm = np.ones_like(V_fdm)  # (Nt+1, Ns+1)
    for i in range(V_fdm.shape[0]):
        for j in range(V_fdm.shape[1]):
            if payoff_fdm[j] > 0 and abs(V_fdm[i, j] - payoff_fdm[j]) / cfg.K < tol:
                phi_fdm[i, j] = 0.0  # 行权区
            elif payoff_fdm[j] == 0 and V_fdm[i, j] < tol * cfg.K:
                phi_fdm[i, j] = 0.0  # 深度虚值但V≈0的区域 (非行权区, 但可标1)
                phi_fdm[i, j] = 1.0  # 这里其实是继续区
    
    # 简化: 更直接的判断
    phi_fdm = np.ones_like(V_fdm)
    for i in range(V_fdm.shape[0]):
        diff = V_fdm[i, :] - payoff_fdm
        # 行权区: diff < small threshold (FDM 数值精度)
        exercise_mask = diff < 1e-3 * cfg.K  # 差值小于 0.1 元 (K=100)
        # 但还要排除 payoff=0 的区域 (S > K, OTM, 这不是行权区)
        in_the_money = payoff_fdm > 0
        phi_fdm[i, exercise_mask & in_the_money] = 0.0
    
    # 现在 phi_fdm 在 FDM 精细网格上是准确的 0/1
    exercise_pct = (phi_fdm == 0).mean()
    print(f"FDM 行权区比例: {exercise_pct:.4f}")
    
    # 插值到 PINN 网格 (使用最近邻以保持 0/1 不被模糊)
    from scipy.interpolate import RegularGridInterpolator
    phi_fdm_interp = RegularGridInterpolator(
        (t_fdm, S_fdm), phi_fdm,
        method='nearest', bounds_error=False, fill_value=1.0
    )
    
    S_pinn = np.linspace(0, cfg.S_max, cfg.grid_S)
    t_pinn = np.linspace(0, cfg.T, cfg.grid_t)
    tt_pinn, ss_pinn = np.meshgrid(t_pinn, S_pinn, indexing='ij')
    points = np.stack([tt_pinn.ravel(), ss_pinn.ravel()], axis=-1)
    phi_hard = phi_fdm_interp(points).reshape(cfg.grid_t, cfg.grid_S)
    
    # 轻微平滑 (高斯模糊, 模拟 phase-field 的过渡带)
    from scipy.ndimage import gaussian_filter
    sigma_smooth = 1.5  # 像素单位, 约 1-2 个网格点的过渡带
    phi_smooth = gaussian_filter(phi_hard, sigma=sigma_smooth)
    phi_smooth = np.clip(phi_smooth, 0.02, 0.98)  # 避免 BCE 的 log(0)
    
    phi_target = torch.tensor(phi_smooth, dtype=torch.float32).to(cfg.device)
    
    print(f"φ 目标标签统计: mean={phi_target.mean():.4f}, "
          f"min={phi_target.min():.4f}, max={phi_target.max():.4f}, "
          f"std={phi_target.std():.4f}")
    print(f"行权区 (φ<0.1): {(phi_target < 0.1).float().mean():.4f}")
    print(f"继续区 (φ>0.9): {(phi_target > 0.9).float().mean():.4f}")
    # 现在应该: 行权区 > 0.2, 继续区 > 0.4

# ============================================================
# Stage 2: 联合训练 (Phase-Field Energy)
# ============================================================
print("\n" + "=" * 70)
print("Stage 2: 联合训练 (Phase-Field Energy)")
print("=" * 70)

optimizer_mlp = torch.optim.AdamW(v_net.parameters(), lr=cfg.stage2_lr_mlp, weight_decay=1e-5)
optimizer_cnn = torch.optim.AdamW(phi_net.parameters(), lr=cfg.stage2_lr_cnn, weight_decay=1e-5)

scheduler_mlp = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_mlp, T_max=cfg.stage2_epochs, eta_min=1e-6
)
scheduler_cnn = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_cnn, T_max=cfg.stage2_epochs, eta_min=1e-5
)

history_s2 = {'pde': [], 'exercise': [], 'interface': [], 'bc': [], 'total': [], 'rel_err': []}

dS_norm = 1.0 / (cfg.grid_S - 1)
dt_norm = 1.0 / (cfg.grid_t - 1)

# 随机采样的内点数 (每个 epoch 从网格中采样一个子集, 加速计算)
N_sample = min(4096, N_grid)

def focal_bce(pred, target, gamma=2.0):
    """Focal Loss: 对难分类样本加大权重"""
    pred = pred.clamp(1e-6, 1 - 1e-6)
    bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
    # p_t = 分类正确的概率
    p_t = pred * target + (1 - pred) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    return (focal_weight * bce).mean()

for epoch in range(1, cfg.stage2_epochs + 1):
    v_net.train()
    phi_net.train()
    
    # ---- 获取当前 V 预测 ---- 
    # 用网格坐标 (归一化) 创建需要梯度的张量
    S_norm_flat = S_grid.squeeze().reshape(-1, 1).clone().detach().requires_grad_(True)
    t_norm_flat = t_grid.squeeze().reshape(-1, 1).clone().detach().requires_grad_(True)
    
    # 随机采样子集 (可选, 如果网格不大可以用全部)
    if N_grid > N_sample:
        idx = torch.randperm(N_grid, device=cfg.device)[:N_sample]
        S_sample = S_norm_flat[idx]
        t_sample = t_norm_flat[idx]
    else:
        S_sample = S_norm_flat
        t_sample = t_norm_flat
        idx = None
    
    # 计算 V 和 BS 残差
    V_norm_pred, bs_residual = compute_bs_residual(v_net, S_sample, t_sample, cfg)
    # V_norm_pred: (N_sample, 1), bs_residual: (N_sample, 1)
    
    # ---- 获取 φ ----
    # 构建 CNN 输入: 用全网格上的 V_pred (detach) 作为第三通道
    with torch.no_grad():
        S_all = S_grid.squeeze().reshape(-1, 1)
        t_all = t_grid.squeeze().reshape(-1, 1)
        V_all_pred = v_net(S_all, t_all)  # (N_grid, 1)
        V_map = V_all_pred.reshape(1, 1, cfg.grid_t, cfg.grid_S)  # (1,1,H,W)
    
    cnn_input = torch.cat([S_grid, t_grid, V_map], dim=1)  # (1,3,H,W)
    phi_full = phi_net(cnn_input)  # (1,1,H,W)
    phi_2d = phi_full.squeeze()    # (H, W) = (grid_t, grid_S)
    
    # 采样对应的 φ 值
    if idx is not None:
        # idx 对应的 (i_t, i_s) 位置
        i_t = idx // cfg.grid_S
        i_s = idx % cfg.grid_S
        phi_sample = phi_2d[i_t, i_s].unsqueeze(1)  # (N_sample, 1)
    else:
        phi_sample = phi_2d.reshape(-1, 1)
    
    # Payoff (归一化)
    if idx is not None:
        S_orig_sample = S_sample.detach() * cfg.S_max
        payoff_sample = torch.clamp(1.0 - S_orig_sample / cfg.K, min=0.0)
    else:
        payoff_sample = payoff_grid.reshape(-1, 1)
    
    # ---- 计算 Phase-Field Energy Loss ----
    
    # Term 1: PDE 残差 (weighted by φ)
    # E_pde = mean[ φ * |ℒV|² ]
    pde_loss = (phi_sample * bs_residual**2).mean()
    
    # Term 2: 行权约束
    # E_ex = (1/ε) * mean[ (1-φ)² * |V-Ψ|² ]
    exercise_loss = ((1 - phi_sample)**2 * (V_norm_pred - payoff_sample)**2).mean() / cfg.eps
    
    # Term 3: 界面能
    # E_int = mean[ ε|∇φ|² + W(φ)/ε ]
    W_phi = phi_2d**2 * (1 - phi_2d)**2  # 双井势 (H,W)
    grad_phi_sq = phi_gradient_fd(phi_2d, dS_norm, dt_norm)  # (H,W)
    interface_loss = (cfg.eps * grad_phi_sq + W_phi / cfg.eps).mean()
    
    # Term 4: 边界条件 (同 Stage 1)
    n_bc = cfg.grid_t
    S_bc0 = torch.zeros(n_bc, 1, device=cfg.device)
    t_bc = torch.linspace(0, 1, n_bc, device=cfg.device).unsqueeze(1)
    bc_loss_left = F.mse_loss(v_net(S_bc0, t_bc), torch.ones_like(S_bc0))
    
    S_bc1 = torch.ones(n_bc, 1, device=cfg.device)
    bc_loss_right = F.mse_loss(v_net(S_bc1, t_bc), torch.zeros_like(S_bc1))
    
    n_tc = cfg.grid_S
    S_tc = torch.linspace(0, 1, n_tc, device=cfg.device).unsqueeze(1)
    t_tc = torch.ones(n_tc, 1, device=cfg.device)
    V_tc_target = torch.clamp(1.0 - S_tc * cfg.S_max / cfg.K, min=0.0)
    bc_loss_terminal = F.mse_loss(v_net(S_tc, t_tc), V_tc_target)
    
    bc_loss = bc_loss_left + bc_loss_right + bc_loss_terminal

    # Term 5: 行权区锚定
    """
    在 V < Ψ 的点上，φ 应该接近 0
    在 V > Ψ 的点上，φ 应该接近 1
    """
    # 用 FDM 解构建软标签
    with torch.no_grad():
        if idx is not None:
            phi_anchor_target = phi_target[i_t, i_s].unsqueeze(1)
        else:
            phi_anchor_target = phi_target.reshape(-1, 1)

    anchor_loss = focal_bce(
        phi_sample.clamp(1e-6, 1 - 1e-6),
        phi_anchor_target,
        gamma=2.0
    )
    
    # Term 6: φ 体积平衡 
    """
    防止 φ 全部塌缩到 0 或 1
    美式看跌期权大约有 30-50% 的域是行权区
    用 FDM 估计行权区面积比例
    """
    with torch.no_grad():
        exercise_ratio = (V_fdm_norm.reshape(-1) - payoff_grid.reshape(-1) < 0.001).float().mean()
    
    phi_mean = phi_2d.mean()
    target_phi_mean = 1.0 - exercise_ratio  # 继续区比例
    balance_loss = (phi_mean - target_phi_mean)**2
    
    
    # 总损失
    total_loss = (cfg.lambda_pde * pde_loss 
                  + cfg.lambda_ex * exercise_loss 
                  + cfg.lambda_int * interface_loss 
                  + cfg.lambda_bc * bc_loss
                  + cfg.lambda_anchor * anchor_loss
                  + cfg.lambda_phi_balance * balance_loss  
                  )
    
    # 反向传播
    optimizer_mlp.zero_grad()
    optimizer_cnn.zero_grad()
    total_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(phi_net.parameters(), 1.0)
    
    optimizer_mlp.step()
    optimizer_cnn.step()
    scheduler_mlp.step()
    scheduler_cnn.step()
    
    # 记录
    with torch.no_grad():
        V_pred_full = v_net(S_all, t_all).reshape(cfg.grid_t, cfg.grid_S) * cfg.K
        rel_err = torch.norm(V_pred_full - V_fdm_grid) / (torch.norm(V_fdm_grid) + 1e-8)
    
    history_s2['pde'].append(pde_loss.item())
    history_s2['exercise'].append(exercise_loss.item())
    history_s2['interface'].append(interface_loss.item())
    history_s2['bc'].append(bc_loss.item())
    history_s2['total'].append(total_loss.item())
    history_s2['rel_err'].append(rel_err.item())
    
    if epoch % 500 == 0:
        with torch.no_grad():
            p = phi_2d
            print(f"  Epoch {epoch}/{cfg.stage2_epochs} | "
                  f"PDE={pde_loss.item():.6f} | Ex={exercise_loss.item():.6f} | "
                  f"Int={interface_loss.item():.6f} | Anchor={anchor_loss.item():.6f} | "
                  f"BC={bc_loss.item():.6f} | Total={total_loss.item():.6f} | "
                  f"RelErr={rel_err.item():.4f} | "
                  f"φ [mean={p.mean():.3f}, std={p.std():.3f}, "
                  f"min={p.min():.3f}, max={p.max():.3f}]")


# ============================================================
# Stage 3: L-BFGS 精调 (仅 MLP, 固定 φ)
# ============================================================
print("\n" + "=" * 70)
print("Stage 3: L-BFGS 精调")
print("=" * 70)

phi_net.eval()
for p in phi_net.parameters():
    p.requires_grad_(False)

# 固定 φ map
with torch.no_grad():
    S_all = S_grid.squeeze().reshape(-1, 1)
    t_all = t_grid.squeeze().reshape(-1, 1)
    V_for_cnn = v_net(S_all, t_all).reshape(1, 1, cfg.grid_t, cfg.grid_S)
    cnn_input_fixed = torch.cat([S_grid, t_grid, V_for_cnn], dim=1)
    phi_fixed = phi_net(cnn_input_fixed).squeeze()  # (H, W)

lbfgs_optimizer = torch.optim.LBFGS(
    v_net.parameters(), lr=0.5, max_iter=5, 
    history_size=20, line_search_fn='strong_wolfe'
)

history_s3 = {'total': [], 'rel_err': []}

for step in range(cfg.stage3_steps):
    def closure():
        lbfgs_optimizer.zero_grad()
        
        S_n = S_grid.squeeze().reshape(-1, 1).clone().detach().requires_grad_(True)
        t_n = t_grid.squeeze().reshape(-1, 1).clone().detach().requires_grad_(True)
        
        V_norm, bs_res = compute_bs_residual(v_net, S_n, t_n, cfg)
        phi_flat = phi_fixed.reshape(-1, 1)
        payoff_flat = payoff_grid.reshape(-1, 1)
        
        pde = (phi_flat * bs_res**2).mean()
        ex = ((1 - phi_flat)**2 * (V_norm - payoff_flat)**2).mean() / cfg.eps
        
        # BC
        n_bc = cfg.grid_t
        t_bc = torch.linspace(0, 1, n_bc, device=cfg.device).unsqueeze(1)
        bc_l = F.mse_loss(v_net(torch.zeros(n_bc, 1, device=cfg.device), t_bc), 
                          torch.ones(n_bc, 1, device=cfg.device))
        bc_r = F.mse_loss(v_net(torch.ones(n_bc, 1, device=cfg.device), t_bc),
                          torch.zeros(n_bc, 1, device=cfg.device))
        n_tc = cfg.grid_S
        S_tc = torch.linspace(0, 1, n_tc, device=cfg.device).unsqueeze(1)
        bc_t = F.mse_loss(v_net(S_tc, torch.ones(n_tc, 1, device=cfg.device)),
                          torch.clamp(1.0 - S_tc * cfg.S_max / cfg.K, min=0.0))
        bc = bc_l + bc_r + bc_t
        
        loss = cfg.lambda_pde * pde + cfg.lambda_ex * ex + cfg.lambda_bc * bc
        loss.backward()
        return loss
    
    loss_val = lbfgs_optimizer.step(closure)
    
    with torch.no_grad():
        V_pred_full = v_net(S_all, t_all).reshape(cfg.grid_t, cfg.grid_S) * cfg.K
        rel_err = torch.norm(V_pred_full - V_fdm_grid) / (torch.norm(V_fdm_grid) + 1e-8)
    
    history_s3['total'].append(loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val)
    history_s3['rel_err'].append(rel_err.item())
    
    if (step + 1) % 10 == 0:
        print(f"  Step {step+1}/{cfg.stage3_steps} | Loss={loss_val:.6f} | RelErr={rel_err.item():.4f}")


# ============================================================
# 保存结果
# ============================================================
print("\n保存结果...")

# 最终预测
with torch.no_grad():
    S_all = S_grid.squeeze().reshape(-1, 1)
    t_all = t_grid.squeeze().reshape(-1, 1)
    V_final_norm = v_net(S_all, t_all).reshape(cfg.grid_t, cfg.grid_S)
    V_final = (V_final_norm * cfg.K).cpu().numpy()
    
    V_for_cnn = V_final_norm.reshape(1, 1, cfg.grid_t, cfg.grid_S)
    cnn_input_final = torch.cat([S_grid, t_grid, V_for_cnn], dim=1)
    # 需要重新启用 phi_net
    for p in phi_net.parameters():
        p.requires_grad_(False)  # 保持冻结
    phi_net.eval()
    phi_final = phi_net(cnn_input_final).squeeze().cpu().numpy()

S_save = np.linspace(0, cfg.S_max, cfg.grid_S)
t_save = np.linspace(0, cfg.T, cfg.grid_t)

np.savez('scheme1_results.npz',
         S=S_save, t=t_save,
         V_pred=V_final,
         V_fdm=V_fdm_grid.cpu().numpy(),
         phi=phi_final,
         fdm_boundary=fdm_boundary,
         S_fdm=S_fdm, t_fdm=t_fdm)

# 保存训练历史
history = {
    'stage1': {k: [float(v) for v in vals] for k, vals in history_s1.items()},
    'stage2': {k: [float(v) for v in vals] for k, vals in history_s2.items()},
    'stage3': {k: [float(v) for v in vals] for k, vals in history_s3.items()},
}
with open('scheme1_history.json', 'w') as f:
    json.dump(history, f)

# 保存模型
torch.save({
    'v_net': v_net.state_dict(),
    'phi_net': phi_net.state_dict(),
    'config': {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
}, 'scheme1_best.pth')

print("所有结果已保存!")
print(f"  scheme1_results.npz")
print(f"  scheme1_history.json")
print(f"  scheme1_best.pth")
print(f"\n最终 V RelErr = {history_s3['rel_err'][-1]:.6f}" if history_s3['rel_err'] else "")
