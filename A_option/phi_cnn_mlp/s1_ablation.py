"""
scheme1_ablation.py
===================
Phase-Field + PINN 美式看跌期权 —— 消融实验
对比 5 种变体，评估各组件的贡献。

变体:
  A) Full model          (完整模型，读取已有结果)
  B) No Phase-Field      (λ_int=0, 去掉界面能)
  C) No Anchor           (λ_anchor=0, 纯 Phase-Field 能量驱动 φ)
  D) MLP-MLP             (用 MLP 替代 CNN 预测 φ)
  E) Pure Penalty PINN   (经典惩罚法，不用 φ)

输出:
  ablation_results.json   —— 各变体的定量指标
  ablation_results.npz    —— 各变体的 V_pred, phi 等数据
  figures/fig_ablation_*.png/pdf —— 对比图表
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import os
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

# ============================================================
# 0. 配置
# ============================================================
class Config:
    # 期权参数
    S_max = 200.0
    T = 1.0
    K = 100.0
    r = 0.05
    sigma = 0.2

    # 网格
    grid_S = 128
    grid_t = 64

    # FDM
    fdm_S = 500
    fdm_t = 2000

    # Phase-Field
    eps = 0.05

    # MLP (Siren)
    hidden_dim = 256
    num_layers = 4
    omega_0 = 5.0

    # CNN (U-Net)
    cnn_base_ch = 32
    cnn_levels = 3
    cnn_in_channels = 3

    # MLP-phi (变体 D)
    phi_mlp_hidden = 128
    phi_mlp_layers = 4

    # 训练 epochs
    stage1_epochs = 3000
    stage15_epochs = 1500
    stage2_epochs = 20000
    stage3_steps = 50

    # 学习率
    stage1_lr = 2e-3
    stage15_lr = 2e-3
    stage2_lr_mlp = 5e-4
    stage2_lr_cnn = 5e-3
    stage2_lr_phi_mlp = 2e-3

    # 损失权重 (Full model 默认)
    lambda_pde = 1.0
    lambda_ex = 3.0
    lambda_int = 0.01
    lambda_bc = 50.0
    lambda_anchor = 25.0
    lambda_balance = 2.0

    # Anchor 衰减
    anchor_decay_start = 10000
    anchor_decay_end = 20000
    anchor_decay_ratio = 0.3

    # 惩罚法 (变体 E)
    lambda_penalty = 50.0

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印间隔
    log_interval_s1 = 500
    log_interval_s15 = 300
    log_interval_s2 = 1000


cfg = Config()
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. FDM 求解器 (Thomas 算法, fully-implicit)
# ============================================================
def solve_fdm(cfg):
    """全隐式 FDM + Thomas 算法求解美式看跌"""
    Ns = cfg.fdm_S
    Nt = cfg.fdm_t
    S = np.linspace(0, cfg.S_max, Ns + 1)
    dt = cfg.T / Nt
    dS = S[1] - S[0]

    V = np.maximum(cfg.K - S, 0.0).astype(np.float64)
    V_all = np.zeros((Nt + 1, Ns + 1), dtype=np.float64)
    V_all[0, :] = V.copy()

    fb = np.zeros(Nt + 1)
    payoff = np.maximum(cfg.K - S, 0.0)

    # 找初始自由边界
    ex_idx = np.where(np.abs(V - payoff) < 1e-10)[0]
    fb[0] = S[ex_idx[-1]] if len(ex_idx) > 0 else 0.0

    for n in range(1, Nt + 1):
        a = np.zeros(Ns + 1)
        b = np.zeros(Ns + 1)
        c = np.zeros(Ns + 1)
        d = V.copy()

        for j in range(1, Ns):
            sj = S[j]
            alpha = 0.5 * cfg.sigma**2 * sj**2 / dS**2
            beta = cfg.r * sj / (2 * dS)
            a[j] = -(alpha - beta) * dt
            b[j] = 1 + (2 * alpha + cfg.r) * dt
            c[j] = -(alpha + beta) * dt

        # 边界条件
        b[0] = 1.0
        d[0] = cfg.K * np.exp(-cfg.r * n * dt)  # 近似
        b[Ns] = 1.0
        d[Ns] = 0.0

        # Thomas 算法
        for j in range(1, Ns + 1):
            if abs(b[j - 1]) < 1e-15:
                continue
            w = a[j] / b[j - 1]
            b[j] -= w * c[j - 1]
            d[j] -= w * d[j - 1]

        V[Ns] = d[Ns] / b[Ns] if abs(b[Ns]) > 1e-15 else 0.0
        for j in range(Ns - 1, -1, -1):
            V[j] = (d[j] - c[j] * V[j + 1]) / b[j] if abs(b[j]) > 1e-15 else 0.0

        # 美式约束
        V = np.maximum(V, payoff)
        V_all[n, :] = V.copy()

        # 自由边界
        ex_mask = np.abs(V - payoff) < 1e-6
        ex_idx = np.where(ex_mask & (payoff > 0))[0]
        fb[n] = S[ex_idx[-1]] if len(ex_idx) > 0 else fb[n - 1]

    return S, V_all, fb


# ============================================================
# 2. 构建 φ 目标标签 (signed distance + sigmoid)
# ============================================================
def build_phi_target(V_fdm, S_fdm, t_fdm, S_pinn, t_pinn, cfg):
    """在 FDM 网格构建硬标签，插值到 PINN 网格，距离变换平滑"""
    payoff_fdm = np.maximum(cfg.K - S_fdm, 0.0)
    phi_fdm = np.ones_like(V_fdm)
    for i in range(V_fdm.shape[0]):
        diff = V_fdm[i, :] - payoff_fdm
        exercise_mask = (diff < 1e-3 * cfg.K) & (payoff_fdm > 0)
        phi_fdm[i, exercise_mask] = 0.0

    # 最近邻插值到 PINN 网格
    interp_fn = RegularGridInterpolator(
        (t_fdm, S_fdm), phi_fdm, method='nearest', bounds_error=False, fill_value=1.0
    )
    tt_pinn, ss_pinn = np.meshgrid(t_pinn, S_pinn, indexing='ij')
    points = np.stack([tt_pinn.ravel(), ss_pinn.ravel()], axis=-1)
    phi_hard = interp_fn(points).reshape(len(t_pinn), len(S_pinn))

    # 距离变换 + sigmoid 平滑
    continuation = (phi_hard > 0.5).astype(float)
    dist_pos = distance_transform_edt(continuation)
    dist_neg = distance_transform_edt(1 - continuation)
    signed_dist = dist_pos - dist_neg
    transition_width = 1.2
    phi_smooth = 1.0 / (1.0 + np.exp(-signed_dist / (transition_width * 0.5)))
    phi_smooth = np.clip(phi_smooth, 0.02, 0.98)

    return phi_smooth.astype(np.float32)


# ============================================================
# 3. 网络定义
# ============================================================
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, omega_0=5.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_f, 1.0 / in_f)
            else:
                bound = np.sqrt(6.0 / in_f) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ValueMLP(nn.Module):
    """Siren MLP: (S_norm, t_norm) -> V_norm"""
    def __init__(self, hidden_dim=256, num_layers=4, omega_0=5.0):
        super().__init__()
        layers = [SirenLayer(2, hidden_dim, omega_0=omega_0, is_first=True)]
        for _ in range(num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_dim) / omega_0
            self.out.weight.uniform_(-bound, bound)

    def forward(self, S_norm, t_norm):
        x = torch.cat([S_norm, t_norm], dim=-1)
        return self.out(self.net(x))


class PhiUNet(nn.Module):
    """轻量 U-Net: (1, C_in, H, W) -> (1, 1, H, W), sigmoid 输出"""
    def __init__(self, in_channels=3, base_ch=32, levels=3):
        super().__init__()
        self.levels = levels
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        ch = in_channels
        enc_chs = []
        for i in range(levels):
            out_ch = base_ch * (2 ** i)
            self.enc.append(self._block(ch, out_ch))
            enc_chs.append(out_ch)
            ch = out_ch

        self.bottleneck = self._block(ch, ch * 2)
        ch = ch * 2

        for i in range(levels - 1, -1, -1):
            out_ch = enc_chs[i]
            self.dec.append(nn.Sequential(
                nn.ConvTranspose2d(ch, out_ch, 2, stride=2),
                self._block(out_ch * 2, out_ch)
            ))
            ch = out_ch

        self.head = nn.Conv2d(ch, 1, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        skips = []
        h = x
        for enc in self.enc:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        for i, dec in enumerate(self.dec):
            up = dec[0](h)
            skip = skips[self.levels - 1 - i]
            # 处理尺寸不匹配
            if up.shape != skip.shape:
                up = F.interpolate(up, size=skip.shape[2:], mode='bilinear', align_corners=False)
            h = dec[1](torch.cat([up, skip], dim=1))
        return torch.sigmoid(self.head(h))


class PhiMLP(nn.Module):
    """MLP 预测 φ: (S_norm, t_norm) -> φ ∈ [0,1]"""
    def __init__(self, hidden_dim=128, num_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, S_norm, t_norm):
        x = torch.cat([S_norm, t_norm], dim=-1)
        return torch.sigmoid(self.net(x))


# ============================================================
# 4. 工具函数
# ============================================================
def compute_bs_residual(v_net, S_n, t_n, cfg):
    """计算 BS PDE 残差，返回 (V_norm, residual)"""
    V_n = v_net(S_n, t_n)

    dV_dS = torch.autograd.grad(V_n, S_n, torch.ones_like(V_n),
                                 create_graph=True, retain_graph=True)[0]
    dV_dt = torch.autograd.grad(V_n, t_n, torch.ones_like(V_n),
                                 create_graph=True, retain_graph=True)[0]
    d2V_dS2 = torch.autograd.grad(dV_dS, S_n, torch.ones_like(dV_dS),
                                    create_graph=True, retain_graph=True)[0]

    # 反归一化导数
    V = V_n * cfg.K
    dVdS = dV_dS * cfg.K / cfg.S_max
    dVdt = dV_dt * cfg.K / cfg.T
    d2VdS2 = d2V_dS2 * cfg.K / (cfg.S_max ** 2)
    S_real = S_n * cfg.S_max

    residual = dVdt + 0.5 * cfg.sigma**2 * S_real**2 * d2VdS2 + \
               cfg.r * S_real * dVdS - cfg.r * V

    # 归一化残差
    residual_norm = residual / cfg.K

    return V_n, residual_norm


def phi_gradient_fd_inner(phi_2d, dS, dt):
    """有限差分计算 |∇φ|²，仅内部点"""
    dphi_dS = (phi_2d[1:-1, 2:] - phi_2d[1:-1, :-2]) / (2.0 * dS)
    dphi_dt = (phi_2d[2:, 1:-1] - phi_2d[:-2, 1:-1]) / (2.0 * dt)
    return dphi_dS**2 + dphi_dt**2


def focal_bce(pred, target, gamma=2.0):
    """Focal BCE loss"""
    pred = pred.clamp(1e-6, 1 - 1e-6)
    bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
    p_t = pred * target + (1 - pred) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def compute_rel_err(V_pred_norm, V_fdm_norm):
    """计算相对 L2 误差"""
    num = torch.sqrt(torch.mean((V_pred_norm - V_fdm_norm) ** 2))
    den = torch.sqrt(torch.mean(V_fdm_norm ** 2)) + 1e-12
    return (num / den).item()


# ============================================================
# 5. 准备数据 (所有变体共用)
# ============================================================
def prepare_data(cfg):
    """FDM 求解 + 构建训练数据"""
    print("=" * 70)
    print("求解 FDM 参考解...")
    t0 = time.time()
    S_fdm, V_fdm_all, fb_fdm = solve_fdm(cfg)
    t_fdm = np.linspace(0, cfg.T, cfg.fdm_t + 1)
    print(f"  FDM 耗时: {time.time() - t0:.1f}s, shape={V_fdm_all.shape}")

    # PINN 网格
    S_pinn = np.linspace(0, cfg.S_max, cfg.grid_S)
    t_pinn = np.linspace(0, cfg.T, cfg.grid_t)

    # FDM 插值到 PINN 网格
    interp_V = RegularGridInterpolator(
        (t_fdm, S_fdm), V_fdm_all, method='linear', bounds_error=False, fill_value=0.0
    )
    tt_p, ss_p = np.meshgrid(t_pinn, S_pinn, indexing='ij')
    pts = np.stack([tt_p.ravel(), ss_p.ravel()], axis=-1)
    V_fdm_pinn = interp_V(pts).reshape(len(t_pinn), len(S_pinn)).astype(np.float32)

    # 自由边界插值
    from scipy.interpolate import interp1d
    fb_interp_fn = interp1d(t_fdm, fb_fdm, kind='linear', fill_value='extrapolate')
    fb_pinn = fb_interp_fn(t_pinn).astype(np.float32)

    # φ 目标标签
    phi_target = build_phi_target(V_fdm_all, S_fdm, t_fdm, S_pinn, t_pinn, cfg)

    # 归一化
    V_fdm_norm = V_fdm_pinn / cfg.K

    # 转为 tensor
    device = cfg.device
    S_t = torch.tensor(S_pinn, dtype=torch.float32, device=device)
    t_t = torch.tensor(t_pinn, dtype=torch.float32, device=device)
    V_fdm_norm_t = torch.tensor(V_fdm_norm, dtype=torch.float32, device=device)
    phi_target_t = torch.tensor(phi_target, dtype=torch.float32, device=device)

    # 网格张量
    ss, tt = torch.meshgrid(t_t, S_t, indexing='ij')
    S_flat = (ss.reshape(-1, 1) / cfg.S_max)  # 注意: 这是 S_norm 还是 t_norm 取决于 meshgrid 顺序
    # 修正: meshgrid(t, S, indexing='ij') -> ss 是 t 的广播, tt 是 S 的广播
    # 重新命名以避免混淆
    tt_grid, ss_grid = torch.meshgrid(t_t, S_t, indexing='ij')
    S_norm_flat = (ss_grid / cfg.S_max).reshape(-1, 1)
    t_norm_flat = (tt_grid / cfg.T).reshape(-1, 1)

    # CNN 输入网格
    S_grid_2d = (ss_grid / cfg.S_max).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t_grid_2d = (tt_grid / cfg.T).unsqueeze(0).unsqueeze(0)      # (1,1,H,W)

    # payoff 归一化
    payoff_grid = torch.clamp(cfg.K - ss_grid, min=0.0) / cfg.K  # (grid_t, grid_S)

    # 行权区比例
    exercise_ratio = (phi_target_t < 0.1).float().mean().item()

    data_dict = {
        'S_pinn': S_pinn, 't_pinn': t_pinn,
        'V_fdm_pinn': V_fdm_pinn, 'V_fdm_norm': V_fdm_norm,
        'phi_target': phi_target, 'fb_pinn': fb_pinn,
        'V_fdm_norm_t': V_fdm_norm_t, 'phi_target_t': phi_target_t,
        'S_norm_flat': S_norm_flat, 't_norm_flat': t_norm_flat,
        'S_grid_2d': S_grid_2d, 't_grid_2d': t_grid_2d,
        'payoff_grid': payoff_grid,
        'exercise_ratio': exercise_ratio,
        'ss_grid': ss_grid, 'tt_grid': tt_grid,
    }
    return data_dict


# ============================================================
# 6. Stage 1: MLP 预热 (所有变体共用)
# ============================================================
def train_stage1(v_net, data, cfg):
    """FDM 监督预训练 MLP"""
    print("\n" + "=" * 70)
    print("Stage 1: MLP Pre-training (FDM supervision)")
    print("=" * 70)

    V_target = data['V_fdm_norm_t'].reshape(-1, 1)
    S_n = data['S_norm_flat']
    t_n = data['t_norm_flat']

    optimizer = torch.optim.AdamW(v_net.parameters(), lr=cfg.stage1_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.stage1_epochs, eta_min=1e-5)

    history = {'mse': [], 'rel_err': []}

    for epoch in range(1, cfg.stage1_epochs + 1):
        v_net.train()
        V_pred = v_net(S_n, t_n)
        loss_mse = F.mse_loss(V_pred, V_target)

        # BC: V(S_max, t) = 0
        bc_idx = (S_n > 0.99).squeeze()
        if bc_idx.sum() > 0:
            loss_bc = F.mse_loss(V_pred[bc_idx], torch.zeros_like(V_pred[bc_idx]))
        else:
            loss_bc = torch.tensor(0.0, device=cfg.device)

        loss = loss_mse + cfg.lambda_bc * loss_bc

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % cfg.log_interval_s1 == 0:
            with torch.no_grad():
                rel_err = compute_rel_err(V_pred, V_target)
                history['mse'].append(loss_mse.item())
                history['rel_err'].append(rel_err)
                print(f"  Epoch {epoch}/{cfg.stage1_epochs} | "
                      f"MSE={loss_mse.item():.6e} | BC={loss_bc.item():.6e} | "
                      f"RelErr={rel_err:.6f}")

    with torch.no_grad():
        V_final = v_net(S_n, t_n)
        rel_err = compute_rel_err(V_final, V_target)
        print(f"\n  Stage 1 完成: RelErr={rel_err:.6f}")

    return history


# ============================================================
# 7. Stage 1.5: φ Warm-Start (CNN / MLP-φ)
# ============================================================
def train_stage15_cnn(phi_net, v_net, data, cfg):
    """用 BCE 训练 CNN 拟合 φ 目标标签"""
    print("\n" + "=" * 70)
    print("Stage 1.5: φ Warm-Start (CNN)")
    print("=" * 70)

    phi_target_4d = data['phi_target_t'].unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        V_map = v_net(data['S_norm_flat'], data['t_norm_flat']).reshape(
            1, 1, cfg.grid_t, cfg.grid_S)
    cnn_input = torch.cat([data['S_grid_2d'], data['t_grid_2d'], V_map.detach()], dim=1)

    optimizer = torch.optim.Adam(phi_net.parameters(), lr=cfg.stage15_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.stage15_epochs, eta_min=1e-5)

    history = {'bce': []}

    for epoch in range(1, cfg.stage15_epochs + 1):
        phi_net.train()
        phi_pred = phi_net(cnn_input)
        loss = F.binary_cross_entropy(phi_pred, phi_target_4d)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % cfg.log_interval_s15 == 0:
            with torch.no_grad():
                p = phi_pred.squeeze()
                history['bce'].append(loss.item())
                print(f"  Epoch {epoch}/{cfg.stage15_epochs} | "
                      f"BCE={loss.item():.6f} | "
                      f"φ [mean={p.mean():.4f}, std={p.std():.4f}] | "
                      f"φ<0.1: {(p < 0.1).float().mean():.3f} | "
                      f"φ>0.9: {(p > 0.9).float().mean():.3f}")

    return history


def train_stage15_mlp_phi(phi_mlp, data, cfg):
    """用 BCE 训练 MLP-φ 拟合目标标签"""
    print("\n" + "=" * 70)
    print("Stage 1.5: φ Warm-Start (MLP-φ)")
    print("=" * 70)

    phi_target_flat = data['phi_target_t'].reshape(-1, 1)
    S_n = data['S_norm_flat']
    t_n = data['t_norm_flat']

    optimizer = torch.optim.Adam(phi_mlp.parameters(), lr=cfg.stage15_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.stage15_epochs, eta_min=1e-5)

    history = {'bce': []}

    for epoch in range(1, cfg.stage15_epochs + 1):
        phi_mlp.train()
        phi_pred = phi_mlp(S_n, t_n)
        loss = F.binary_cross_entropy(phi_pred, phi_target_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % cfg.log_interval_s15 == 0:
            with torch.no_grad():
                p = phi_pred.squeeze()
                history['bce'].append(loss.item())
                print(f"  Epoch {epoch}/{cfg.stage15_epochs} | "
                      f"BCE={loss.item():.6f} | "
                      f"φ [mean={p.mean():.4f}, std={p.std():.4f}] | "
                      f"φ<0.1: {(p < 0.1).float().mean():.3f} | "
                      f"φ>0.9: {(p > 0.9).float().mean():.3f}")

    return history


# ============================================================
# 8. Stage 2: 联合训练 (Phase-Field 变体 B/C/D)
# ============================================================
def train_stage2_phasefield(v_net, phi_net_or_mlp, data, cfg,
                             use_cnn=True,
                             lambda_int_override=None,
                             lambda_anchor_override=None,
                             variant_name="Full"):
    """
    Stage 2 联合训练。
    use_cnn=True  -> phi_net_or_mlp 是 CNN (PhiUNet)
    use_cnn=False -> phi_net_or_mlp 是 MLP (PhiMLP)
    """
    print(f"\n{'=' * 70}")
    print(f"Stage 2: Joint Training — {variant_name}")
    print(f"{'=' * 70}")

    lam_int = lambda_int_override if lambda_int_override is not None else cfg.lambda_int
    lam_anc = lambda_anchor_override if lambda_anchor_override is not None else cfg.lambda_anchor

    S_n_all = data['S_norm_flat'].detach().requires_grad_(True)
    t_n_all = data['t_norm_flat'].detach().requires_grad_(True)
    phi_target_t = data['phi_target_t']
    payoff_grid = data['payoff_grid']
    exercise_ratio = data['exercise_ratio']
    target_phi_mean = 1.0 - exercise_ratio

    dS = 1.0 / cfg.grid_S
    dt = 1.0 / cfg.grid_t

    # 优化器
    if use_cnn:
        opt_mlp = torch.optim.AdamW(v_net.parameters(), lr=cfg.stage2_lr_mlp, weight_decay=1e-5)
        opt_phi = torch.optim.AdamW(phi_net_or_mlp.parameters(), lr=cfg.stage2_lr_cnn, weight_decay=1e-5)
    else:
        opt_mlp = torch.optim.AdamW(v_net.parameters(), lr=cfg.stage2_lr_mlp, weight_decay=1e-5)
        opt_phi = torch.optim.AdamW(phi_net_or_mlp.parameters(), lr=cfg.stage2_lr_phi_mlp, weight_decay=1e-5)

    sch_mlp = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mlp, T_max=cfg.stage2_epochs, eta_min=1e-6)
    sch_phi = torch.optim.lr_scheduler.CosineAnnealingLR(opt_phi, T_max=cfg.stage2_epochs, eta_min=1e-6)

    history = {'pde': [], 'exercise': [], 'interface': [], 'anchor': [],
               'bc': [], 'total': [], 'rel_err': []}

    for epoch in range(1, cfg.stage2_epochs + 1):
        v_net.train()
        phi_net_or_mlp.train()

        # Anchor 权重衰减
        if epoch < cfg.anchor_decay_start:
            anchor_w = lam_anc
        elif epoch < cfg.anchor_decay_end:
            frac = (epoch - cfg.anchor_decay_start) / (cfg.anchor_decay_end - cfg.anchor_decay_start)
            anchor_w = lam_anc * (1.0 - frac * (1.0 - cfg.anchor_decay_ratio))
        else:
            anchor_w = lam_anc * cfg.anchor_decay_ratio

        # 前向: V
        S_n = data['S_norm_flat'].detach().requires_grad_(True)
        t_n = data['t_norm_flat'].detach().requires_grad_(True)
        V_norm_pred, bs_residual = compute_bs_residual(v_net, S_n, t_n, cfg)

        # 前向: φ
        if use_cnn:
            V_map = V_norm_pred.detach().reshape(1, 1, cfg.grid_t, cfg.grid_S)
            cnn_input = torch.cat([data['S_grid_2d'], data['t_grid_2d'], V_map], dim=1)
            phi_2d = phi_net_or_mlp(cnn_input).squeeze(0).squeeze(0)  # (H, W)
        else:
            phi_flat = phi_net_or_mlp(S_n.detach(), t_n.detach())
            phi_2d = phi_flat.reshape(cfg.grid_t, cfg.grid_S)

        phi_flat_all = phi_2d.reshape(-1, 1)

        # --- PDE loss ---
        pde_loss = torch.mean(phi_flat_all * bs_residual ** 2)

        # --- Exercise loss ---
        payoff_flat = payoff_grid.reshape(-1, 1)
        ex_loss = torch.mean((1 - phi_flat_all) ** 2 * (V_norm_pred.detach() - payoff_flat) ** 2 / cfg.eps)

        # --- Interface loss (内部点) ---
        grad_phi_sq = phi_gradient_fd_inner(phi_2d, dS, dt)
        phi_inner = phi_2d[1:-1, 1:-1]
        W_phi = phi_inner ** 2 * (1 - phi_inner) ** 2
        interface_loss = torch.mean(cfg.eps * grad_phi_sq + W_phi / cfg.eps)

        # --- Anchor loss ---
        if lam_anc > 0:
            phi_target_flat = phi_target_t.reshape(-1, 1)
            anchor_loss = focal_bce(phi_flat_all, phi_target_flat, gamma=2.0)
        else:
            anchor_loss = torch.tensor(0.0, device=cfg.device)

        # --- Balance loss ---
        balance_loss = (phi_2d.mean() - target_phi_mean) ** 2

        # --- BC loss ---
        bc_mask = (S_n > 0.99).squeeze()
        if bc_mask.sum() > 0:
            bc_loss = F.mse_loss(V_norm_pred[bc_mask], torch.zeros_like(V_norm_pred[bc_mask]))
        else:
            bc_loss = torch.tensor(0.0, device=cfg.device)

        # --- Total ---
        total_loss = (cfg.lambda_pde * pde_loss +
                      cfg.lambda_ex * ex_loss +
                      lam_int * interface_loss +
                      anchor_w * anchor_loss +
                      cfg.lambda_balance * balance_loss +
                      cfg.lambda_bc * bc_loss)

        opt_mlp.zero_grad()
        opt_phi.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(phi_net_or_mlp.parameters(), 1.0)
        opt_mlp.step()
        opt_phi.step()
        sch_mlp.step()
        sch_phi.step()

        if epoch % cfg.log_interval_s2 == 0:
            with torch.no_grad():
                rel_err = compute_rel_err(V_norm_pred, data['V_fdm_norm_t'].reshape(-1, 1))
                history['pde'].append(pde_loss.item())
                history['exercise'].append(ex_loss.item())
                history['interface'].append(interface_loss.item())
                history['anchor'].append(anchor_loss.item())
                history['bc'].append(bc_loss.item())
                history['total'].append(total_loss.item())
                history['rel_err'].append(rel_err)

                p = phi_2d
                print(f"  Epoch {epoch}/{cfg.stage2_epochs} | "
                      f"PDE={pde_loss.item():.2e} Ex={ex_loss.item():.2e} "
                      f"Int={interface_loss.item():.4f} Anc={anchor_loss.item():.4e} "
                      f"BC={bc_loss.item():.2e} | "
                      f"Total={total_loss.item():.4f} RelErr={rel_err:.6f} | "
                      f"φ[{p.mean():.3f}±{p.std():.3f}]")

    return history


# ============================================================
# 9. Stage 2E: 纯惩罚法 PINN (变体 E, 无 φ)
# ============================================================
def train_stage2_penalty(v_net, data, cfg):
    """经典惩罚法 PINN: L = PDE² + λ_penalty * max(Ψ-V, 0)² + BC"""
    print(f"\n{'=' * 70}")
    print("Stage 2: Pure Penalty PINN (no φ)")
    print(f"{'=' * 70}")

    optimizer = torch.optim.AdamW(v_net.parameters(), lr=cfg.stage2_lr_mlp, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.stage2_epochs, eta_min=1e-6)

    payoff_flat = data['payoff_grid'].reshape(-1, 1)
    history = {'pde': [], 'penalty': [], 'bc': [], 'total': [], 'rel_err': []}

    for epoch in range(1, cfg.stage2_epochs + 1):
        v_net.train()

        S_n = data['S_norm_flat'].detach().requires_grad_(True)
        t_n = data['t_norm_flat'].detach().requires_grad_(True)
        V_norm_pred, bs_residual = compute_bs_residual(v_net, S_n, t_n, cfg)

        # PDE loss
        pde_loss = torch.mean(bs_residual ** 2)

        # Penalty: max(Ψ - V, 0)²
        violation = F.relu(payoff_flat - V_norm_pred)
        penalty_loss = torch.mean(violation ** 2)

        # BC
        bc_mask = (S_n > 0.99).squeeze()
        if bc_mask.sum() > 0:
            bc_loss = F.mse_loss(V_norm_pred[bc_mask], torch.zeros_like(V_norm_pred[bc_mask]))
        else:
            bc_loss = torch.tensor(0.0, device=cfg.device)

        total_loss = cfg.lambda_pde * pde_loss + cfg.lambda_penalty * penalty_loss + cfg.lambda_bc * bc_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % cfg.log_interval_s2 == 0:
            with torch.no_grad():
                rel_err = compute_rel_err(V_norm_pred, data['V_fdm_norm_t'].reshape(-1, 1))
                history['pde'].append(pde_loss.item())
                history['penalty'].append(penalty_loss.item())
                history['bc'].append(bc_loss.item())
                history['total'].append(total_loss.item())
                history['rel_err'].append(rel_err)

                print(f"  Epoch {epoch}/{cfg.stage2_epochs} | "
                      f"PDE={pde_loss.item():.2e} Pen={penalty_loss.item():.2e} "
                      f"BC={bc_loss.item():.2e} | "
                      f"Total={total_loss.item():.4f} RelErr={rel_err:.6f}")

    return history


# ============================================================
# 10. Stage 3: L-BFGS 精调
# ============================================================
def train_stage3(v_net, data, cfg, use_phi=True, phi_net_or_mlp=None, use_cnn=True):
    """L-BFGS 精调 MLP"""
    print(f"\n{'=' * 70}")
    print("Stage 3: L-BFGS Fine-tuning")
    print(f"{'=' * 70}")

    payoff_flat = data['payoff_grid'].reshape(-1, 1)

    # 如果有 φ，预计算并固定
    phi_fixed = None
    if use_phi and phi_net_or_mlp is not None:
        with torch.no_grad():
            if use_cnn:
                V_map = v_net(data['S_norm_flat'], data['t_norm_flat']).reshape(
                    1, 1, cfg.grid_t, cfg.grid_S)
                cnn_input = torch.cat([data['S_grid_2d'], data['t_grid_2d'], V_map], dim=1)
                phi_fixed = phi_net_or_mlp(cnn_input).squeeze().reshape(-1, 1)
            else:
                phi_fixed = phi_net_or_mlp(
                    data['S_norm_flat'], data['t_norm_flat']).detach()

    lbfgs = torch.optim.LBFGS(v_net.parameters(), lr=1.0, max_iter=5,
                                history_size=50, line_search_fn='strong_wolfe')

    history = {'loss': [], 'rel_err': []}

    for step in range(1, cfg.stage3_steps + 1):
        def closure():
            lbfgs.zero_grad()
            S_n = data['S_norm_flat'].detach().requires_grad_(True)
            t_n = data['t_norm_flat'].detach().requires_grad_(True)
            V_norm_pred, bs_residual = compute_bs_residual(v_net, S_n, t_n, cfg)

            if phi_fixed is not None:
                pde_loss = torch.mean(phi_fixed * bs_residual ** 2)
                ex_loss = torch.mean((1 - phi_fixed) ** 2 *
                                     (V_norm_pred - payoff_flat) ** 2 / cfg.eps)
            else:
                pde_loss = torch.mean(bs_residual ** 2)
                ex_loss = cfg.lambda_penalty * torch.mean(
                    F.relu(payoff_flat - V_norm_pred) ** 2)

            bc_mask = (S_n > 0.99).squeeze()
            if bc_mask.sum() > 0:
                bc_loss = F.mse_loss(V_norm_pred[bc_mask],
                                     torch.zeros_like(V_norm_pred[bc_mask]))
            else:
                bc_loss = torch.tensor(0.0, device=cfg.device)

            loss = cfg.lambda_pde * pde_loss + cfg.lambda_ex * ex_loss + cfg.lambda_bc * bc_loss
            loss.backward()
            return loss

        loss_val = lbfgs.step(closure)

        if step % 10 == 0:
            with torch.no_grad():
                V_pred = v_net(data['S_norm_flat'], data['t_norm_flat'])
                rel_err = compute_rel_err(V_pred, data['V_fdm_norm_t'].reshape(-1, 1))
                history['loss'].append(loss_val.item())
                history['rel_err'].append(rel_err)
                print(f"  Step {step}/{cfg.stage3_steps} | "
                      f"Loss={loss_val.item():.6e} | RelErr={rel_err:.6f}")

    return history


# ============================================================
# 11. 提取最终结果
# ============================================================
def extract_results(v_net, data, cfg, phi_net_or_mlp=None, use_cnn=True):
    """提取 V_pred, phi 等最终结果"""
    with torch.no_grad():
        V_pred_norm = v_net(data['S_norm_flat'], data['t_norm_flat'])
        V_pred = (V_pred_norm * cfg.K).reshape(cfg.grid_t, cfg.grid_S).cpu().numpy()
        V_pred_norm_2d = V_pred_norm.reshape(cfg.grid_t, cfg.grid_S).cpu().numpy()
        V_fdm = data['V_fdm_norm_t'].cpu().numpy() * cfg.K
        rel_err = compute_rel_err(V_pred_norm, data['V_fdm_norm_t'].reshape(-1, 1))

        phi = None
        if phi_net_or_mlp is not None:
            if use_cnn:
                V_map = V_pred_norm.reshape(1, 1, cfg.grid_t, cfg.grid_S)
                cnn_input = torch.cat([data['S_grid_2d'], data['t_grid_2d'], V_map], dim=1)
                phi = phi_net_or_mlp(cnn_input).squeeze().cpu().numpy()
            else:
                phi = phi_net_or_mlp(
                    data['S_norm_flat'], data['t_norm_flat']
                ).reshape(cfg.grid_t, cfg.grid_S).cpu().numpy()

    return {
        'V_pred': V_pred,
        'V_pred_norm': V_pred_norm_2d,
        'V_fdm': V_fdm.reshape(cfg.grid_t, cfg.grid_S),
        'phi': phi,
        'rel_err': rel_err,
    }


# ============================================================
# 12. 提取自由边界 (从 φ 或从 V)
# ============================================================
def extract_free_boundary_from_phi(phi, S, threshold=0.5):
    """从 φ 提取自由边界: 每行找 φ=threshold 的交叉点"""
    fb = np.zeros(phi.shape[0])
    for i in range(phi.shape[0]):
        crossings = np.where(np.diff(np.sign(phi[i, :] - threshold)))[0]
        if len(crossings) > 0:
            idx = crossings[-1]
            s0, s1 = S[idx], S[min(idx + 1, len(S) - 1)]
            p0, p1 = phi[i, idx], phi[i, min(idx + 1, phi.shape[1] - 1)]
            fb[i] = s0 + (threshold - p0) / (p1 - p0 + 1e-12) * (s1 - s0)
        else:
            fb[i] = 0.0
    return fb


def extract_free_boundary_from_V(V_pred, S, K):
    """从 V 提取自由边界: 每行找 V = max(K-S, 0) 的交叉点"""
    payoff = np.maximum(K - S, 0)
    fb = np.zeros(V_pred.shape[0])
    for i in range(V_pred.shape[0]):
        diff = V_pred[i, :] - payoff
        crossings = np.where(np.diff(np.sign(diff)))[0]
        if len(crossings) > 0:
            idx = crossings[-1]
            fb[i] = S[idx]
        else:
            # 找最大的 S 使得 V ≈ payoff
            close = np.where((np.abs(diff) < 0.5) & (payoff > 0))[0]
            fb[i] = S[close[-1]] if len(close) > 0 else 0.0
    return fb


# ============================================================
# 13. 可视化: 消融对比图
# ============================================================
def plot_ablation_results(all_results, data, cfg):
    """生成消融对比图"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from scipy.interpolate import interp1d

    S = data['S_pinn']
    t = data['t_pinn']
    fb_fdm = data['fb_pinn']
    V_fdm = data['V_fdm_pinn']

    variant_names = list(all_results.keys())
    colors = {'A_Full': '#1f77b4', 'B_NoInt': '#ff7f0e',
              'C_NoAnc': '#2ca02c', 'D_MLP': '#d62728', 'E_Penalty': '#9467bd'}

    # ---- 图 A1: 自由边界对比 ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, fb_fdm, 'k-', linewidth=2.5, label='FDM reference')
    for name in variant_names:
        res = all_results[name]
        if res.get('fb_phi') is not None:
            ax.plot(t, res['fb_phi'], '--', color=colors.get(name, 'gray'),
                    linewidth=1.8, label=f'{name} (φ=0.5)')
        if name == 'E_Penalty':
            fb_v = extract_free_boundary_from_V(res['V_pred'], S, cfg.K)
            ax.plot(t, fb_v, ':', color=colors[name], linewidth=1.8,
                    label=f'{name} (V=Ψ)')

    ax.set_xlabel(r'Time $t$', fontsize=13)
    ax.set_ylabel(r'Stock price $S$', fontsize=13)
    ax.set_title('Ablation: Free Boundary Comparison', fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_ablation_free_boundary.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_ablation_free_boundary.png"))
    plt.close(fig)
    print("[Ablation] Free boundary comparison saved.")

    # ---- 图 A2: φ 热力图对比 (2×2) ----
    phi_variants = {k: v for k, v in all_results.items() if v.get('phi') is not None}
    if len(phi_variants) > 0:
        n_phi = len(phi_variants)
        ncols = min(n_phi, 2)
        nrows = (n_phi + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
        if nrows * ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        SS, TT = np.meshgrid(S, t)
        for i, (name, res) in enumerate(phi_variants.items()):
            ax = axes[i]
            im = ax.pcolormesh(SS, TT, res['phi'], cmap='coolwarm',
                               shading='auto', vmin=0, vmax=1)
            ax.plot(fb_fdm, t, 'k-', linewidth=1.5)
            ax.set_title(f'{name}', fontsize=12)
            ax.set_xlabel(r'$S$')
            ax.set_ylabel(r'$t$')
            plt.colorbar(im, ax=ax, label=r'$\phi$')

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(r'Ablation: $\phi$ Heatmaps', fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "fig_ablation_phi_heatmaps.pdf"))
        fig.savefig(os.path.join(OUTPUT_DIR, "fig_ablation_phi_heatmaps.png"))
        plt.close(fig)
        print("[Ablation] φ heatmaps saved.")

    # ---- 图 A3: V 误差热力图对比 ----
    n_var = len(variant_names)
    fig, axes = plt.subplots(1, n_var, figsize=(5 * n_var, 4.5))
    if n_var == 1:
        axes = [axes]
    SS, TT = np.meshgrid(S, t)

    for i, name in enumerate(variant_names):
        res = all_results[name]
        abs_err = np.abs(res['V_pred'] - V_fdm.reshape(cfg.grid_t, cfg.grid_S))
        vmax = np.percentile(abs_err, 95)
        im = axes[i].pcolormesh(SS, TT, abs_err, cmap='hot_r', shading='auto',
                                 vmin=0, vmax=max(vmax, 0.01))
        axes[i].plot(fb_fdm, t, 'k--', linewidth=1.5)
        axes[i].set_title(f'{name}\nRelErr={res["rel_err"]:.4f}', fontsize=11)
        axes[i].set_xlabel(r'$S$')
        axes[i].set_ylabel(r'$t$')
        plt.colorbar(im, ax=axes[i], label='|error|')

    fig.suptitle('Ablation: Absolute Error Maps', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_ablation_error_maps.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_ablation_error_maps.png"))
    plt.close(fig)
    print("[Ablation] Error maps saved.")

    # ---- 图 A4: RelErr 训练曲线对比 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in variant_names:
        res = all_results[name]
        if 'stage2_rel_err' in res and len(res['stage2_rel_err']) > 0:
            epochs = np.arange(1, len(res['stage2_rel_err']) + 1) * cfg.log_interval_s2
            ax.semilogy(epochs, res['stage2_rel_err'], '-', color=colors.get(name, 'gray'),
                        linewidth=1.5, label=name)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Relative Error', fontsize=13)
    ax.set_title('Ablation: Stage 2 Convergence', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_ablation_convergence.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_ablation_convergence.png"))
    plt.close(fig)
    print("[Ablation] Convergence curves saved.")

    # ---- 图 A5: 时间截面对比 (t=0.5T) ----
    t_target = 0.5 * cfg.T
    t_idx = np.argmin(np.abs(t - t_target))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左: V(S)
    ax = axes[0]
    ax.plot(S, V_fdm.reshape(cfg.grid_t, cfg.grid_S)[t_idx, :], 'k-',
            linewidth=2.5, label='FDM')
    for name in variant_names:
        res = all_results[name]
        ax.plot(S, res['V_pred'][t_idx, :], '--', color=colors.get(name, 'gray'),
                linewidth=1.5, label=name)
    payoff = np.maximum(cfg.K - S, 0)
    ax.plot(S, payoff, 'k:', linewidth=1, alpha=0.5, label='Payoff')
    ax.set_xlabel(r'$S$', fontsize=13)
    ax.set_ylabel(r'$V(S, t)$', fontsize=13)
    ax.set_title(f'(a) Value at $t={t[t_idx]:.2f}$', fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)

    # 右: Error
    ax2 = axes[1]
    for name in variant_names:
        res = all_results[name]
        err = res['V_pred'][t_idx, :] - V_fdm.reshape(cfg.grid_t, cfg.grid_S)[t_idx, :]
        ax2.plot(S, err, '-', color=colors.get(name, 'gray'), linewidth=1.5,
                 label=name)
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.axvline(fb_fdm[t_idx], color='k', linestyle=':', alpha=0.5, label='FDM boundary')
    ax2.set_xlabel(r'$S$', fontsize=13)
    ax2.set_ylabel(r'$V_{pred} - V_{FDM}$', fontsize=13)
    ax2.set_title(f'(b) Error at $t={t[t_idx]:.2f}$', fontsize=13)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 200)

    fig.suptitle('Ablation: Time Slice Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_ablation_time_slice.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_ablation_time_slice.png"))
    plt.close(fig)
    print("[Ablation] Time slice comparison saved.")

    # ---- 图 A6: 定量指标汇总表 ----
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    table_data = []
    headers = ['Variant', 'V RelErr', 'φ mean', 'φ std',
               'FB mean|err|', 'FB max|err|']

    for name in variant_names:
        res = all_results[name]
        row = [name, f"{res['rel_err']:.6f}"]
        if res.get('phi') is not None:
            row.append(f"{res['phi'].mean():.4f}")
            row.append(f"{res['phi'].std():.4f}")
        else:
            row.extend(['-', '-'])

        if res.get('fb_phi') is not None:
            fb_err = np.abs(res['fb_phi'] - fb_fdm)
            valid = res['fb_phi'] > 0
            if valid.sum() > 0:
                row.append(f"{fb_err[valid].mean():.2f}")
                row.append(f"{fb_err[valid].max():.2f}")
            else:
                row.extend(['-', '-'])
        else:
            row.extend(['-', '-'])

        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # 着色表头
    for j in range(len(headers)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 交替行颜色
    for i in range(len(table_data)):
        color = '#D9E2F3' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

    fig.suptitle('Ablation Study: Quantitative Summary', fontsize=14, y=0.95)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_ablation_summary_table.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_ablation_summary_table.png"))
    plt.close(fig)
    print("[Ablation] Summary table saved.")


# ============================================================
# 14. 运行单个变体
# ============================================================
def run_variant(variant_key, data, cfg):
    """运行单个消融变体，返回结果字典"""
    print(f"\n{'#' * 70}")
    print(f"# 变体 {variant_key}")
    print(f"{'#' * 70}")

    t_start = time.time()

    if variant_key == 'A_Full':
        # 读取已有结果
        if os.path.exists("scheme1_results.npz"):
            existing = np.load("scheme1_results.npz", allow_pickle=True)
            S = existing['S']
            phi = existing['phi'] if 'phi' in existing else None
            fb_phi = extract_free_boundary_from_phi(phi, S) if phi is not None else None

            # 读取 history
            stage2_rel_err = []
            if os.path.exists("scheme1_history.json"):
                with open("scheme1_history.json", 'r') as f:
                    hist = json.load(f)
                s2 = hist.get('stage2', {})
                stage2_rel_err = s2.get('rel_err', [])

            result = {
                'V_pred': existing['V_pred'],
                'phi': phi,
                'rel_err': float(existing.get('rel_err', 0)),
                'fb_phi': fb_phi,
                'stage2_rel_err': stage2_rel_err,
                'time': 0,
            }
            # 重新计算 rel_err
            if 'V_fdm' in existing:
                V_p = existing['V_pred']
                V_f = existing['V_fdm']
                if V_p.shape == V_f.shape:
                    num = np.sqrt(np.mean((V_p - V_f)**2))
                    den = np.sqrt(np.mean(V_f**2)) + 1e-12
                    result['rel_err'] = num / den

            print(f"  [A_Full] 从已有文件加载, RelErr={result['rel_err']:.6f}")
            return result
        else:
            print("  [A_Full] scheme1_results.npz 不存在，作为完整模型重新训练")
            variant_key = 'A_Full_retrain'

    # 初始化 V 网络
    v_net = ValueMLP(cfg.hidden_dim, cfg.num_layers, cfg.omega_0).to(cfg.device)

    # Stage 1
    s1_hist = train_stage1(v_net, data, cfg)

    if variant_key == 'E_Penalty':
        # 纯惩罚法，无 φ
        s2_hist = train_stage2_penalty(v_net, data, cfg)
        s3_hist = train_stage3(v_net, data, cfg, use_phi=False)
        results = extract_results(v_net, data, cfg)
        results['stage2_rel_err'] = s2_hist['rel_err']
        results['fb_phi'] = None
        results['time'] = time.time() - t_start
        return results

    elif variant_key == 'D_MLP':
        # MLP-φ 替代 CNN
        phi_mlp = PhiMLP(cfg.phi_mlp_hidden, cfg.phi_mlp_layers).to(cfg.device)
        train_stage15_mlp_phi(phi_mlp, data, cfg)
        s2_hist = train_stage2_phasefield(
            v_net, phi_mlp, data, cfg, use_cnn=False, variant_name="D_MLP-MLP")
        s3_hist = train_stage3(v_net, data, cfg, use_phi=True,
                               phi_net_or_mlp=phi_mlp, use_cnn=False)
        results = extract_results(v_net, data, cfg, phi_mlp, use_cnn=False)
        results['stage2_rel_err'] = s2_hist['rel_err']
        results['fb_phi'] = extract_free_boundary_from_phi(results['phi'], data['S_pinn'])
        results['time'] = time.time() - t_start
        return results

    else:
        # CNN 变体 (B_NoInt, C_NoAnc, A_Full_retrain)
        phi_net = PhiUNet(cfg.cnn_in_channels, cfg.cnn_base_ch, cfg.cnn_levels).to(cfg.device)
        train_stage15_cnn(phi_net, v_net, data, cfg)

        lam_int = None
        lam_anc = None

        if variant_key == 'B_NoInt':
            lam_int = 0.0
        elif variant_key == 'C_NoAnc':
            lam_anc = 0.0

        s2_hist = train_stage2_phasefield(
            v_net, phi_net, data, cfg, use_cnn=True,
            lambda_int_override=lam_int,
            lambda_anchor_override=lam_anc,
            variant_name=variant_key)

        s3_hist = train_stage3(v_net, data, cfg, use_phi=True,
                               phi_net_or_mlp=phi_net, use_cnn=True)
        results = extract_results(v_net, data, cfg, phi_net, use_cnn=True)
        results['stage2_rel_err'] = s2_hist['rel_err']
        results['fb_phi'] = extract_free_boundary_from_phi(results['phi'], data['S_pinn'])
        results['time'] = time.time() - t_start
        return results


# ============================================================
# 15. 主程序
# ============================================================
def main():
    print("=" * 70)
    print("Phase-Field + PINN 美式看跌期权 —— 消融实验")
    print("=" * 70)

    # 准备数据
    data = prepare_data(cfg)

    # φ 标签统计
    pt = data['phi_target_t']
    print(f"\nφ 目标标签: mean={pt.mean():.4f}, min={pt.min():.4f}, "
          f"max={pt.max():.4f}, std={pt.std():.4f}")
    print(f"  行权区(φ<0.1): {(pt < 0.1).float().mean():.1%}")
    print(f"  继续区(φ>0.9): {(pt > 0.9).float().mean():.1%}")

    # 运行各变体
    variants = ['A_Full', 'B_NoInt', 'C_NoAnc', 'D_MLP', 'E_Penalty']
    all_results = {}

    for var in variants:
        all_results[var] = run_variant(var, data, cfg)
        print(f"\n  >>> {var} 完成: RelErr={all_results[var]['rel_err']:.6f}, "
              f"耗时={all_results[var]['time']:.1f}s")

    # 打印汇总
    print("\n" + "=" * 70)
    print("消融实验结果汇总")
    print("=" * 70)
    print(f"{'Variant':<15} {'RelErr':>10} {'φ mean':>10} {'φ std':>10} {'Time(s)':>10}")
    print("-" * 55)
    for name, res in all_results.items():
        phi_mean = f"{res['phi'].mean():.4f}" if res.get('phi') is not None else '-'
        phi_std = f"{res['phi'].std():.4f}" if res.get('phi') is not None else '-'
        print(f"{name:<15} {res['rel_err']:>10.6f} {phi_mean:>10} {phi_std:>10} "
              f"{res['time']:>10.1f}")

    # 生成对比图表
    print("\n生成可视化图表...")
    plot_ablation_results(all_results, data, cfg)

    # 保存结果
    save_dict = {}
    for name, res in all_results.items():
        save_dict[f"{name}_V_pred"] = res['V_pred']
        save_dict[f"{name}_rel_err"] = res['rel_err']
        if res.get('phi') is not None:
            save_dict[f"{name}_phi"] = res['phi']
        if res.get('fb_phi') is not None:
            save_dict[f"{name}_fb_phi"] = res['fb_phi']
        if 'stage2_rel_err' in res:
            save_dict[f"{name}_s2_relerr"] = np.array(res['stage2_rel_err'])

    save_dict['S'] = data['S_pinn']
    save_dict['t'] = data['t_pinn']
    save_dict['V_fdm'] = data['V_fdm_pinn']
    save_dict['fb_fdm'] = data['fb_pinn']
    save_dict['phi_target'] = data['phi_target']
    np.savez("ablation_results.npz", **save_dict)

    # 保存 JSON 汇总
    summary = {}
    for name, res in all_results.items():
        entry = {
            'rel_err': float(res['rel_err']),
            'time': float(res['time']),
        }
        if res.get('phi') is not None:
            entry['phi_mean'] = float(res['phi'].mean())
            entry['phi_std'] = float(res['phi'].std())
            entry['phi_min'] = float(res['phi'].min())
            entry['phi_max'] = float(res['phi'].max())
        if res.get('fb_phi') is not None:
            fb_err = np.abs(res['fb_phi'] - data['fb_pinn'])
            valid = res['fb_phi'] > 0
            if valid.sum() > 0:
                entry['fb_mean_err'] = float(fb_err[valid].mean())
                entry['fb_max_err'] = float(fb_err[valid].max())
        summary[name] = entry

    with open("ablation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("消融实验完成！")
    print(f"  数据: ablation_results.npz")
    print(f"  汇总: ablation_results.json")
    print(f"  图表: {OUTPUT_DIR}/fig_ablation_*.pdf/png")
    print("=" * 70)


if __name__ == "__main__":
    main()
