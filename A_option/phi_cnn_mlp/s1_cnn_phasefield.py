"""
scheme1_cnn_phasefield.py
方案1: CNN学习φ + MLP学习V (Phase-Field Energy + Anchor)
美式看跌期权自由边界求解

修正版: 
  1. autograd 求导确认正确
  2. 界面能只在内部点计算, 避免边界伪梯度
  3. 消除重复前向传播, 复用 V_norm_pred
  4. φ 标签用距离变换+sigmoid 替代高斯平滑
  5. EMA (Exponential Moving Average) for v_net in Stage 2
  6. 分阶段计时器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt


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
    eps = 0.05
    lambda_int = 0.01
    lambda_ex = 3.0
    lambda_pde = 1.0
    lambda_bc = 50.0
    lambda_anchor = 25.0
    lambda_balance = 2.0

    # 网格
    grid_S = 128
    grid_t = 64

    # MLP (SIREN)
    mlp_hidden = 256
    mlp_layers = 4
    omega_0 = 5.0

    # CNN (U-Net)
    cnn_base_ch = 32
    cnn_levels = 3
    cnn_in_channels = 3  # (S_norm, t_norm, V_pred)

    # FDM
    fdm_S = 500
    fdm_t = 2000

    # 训练
    stage1_epochs = 3000
    stage1_lr = 2e-3
    stage15_epochs = 1500
    stage15_lr = 2e-3
    stage2_epochs = 20000
    stage2_lr_mlp = 5e-4
    stage2_lr_cnn = 5e-3
    stage3_steps = 50

    # anchor 衰减
    anchor_decay_start = 10000
    anchor_decay_end = 20000
    anchor_min_ratio = 0.3

    # EMA
    ema_decay = 0.999

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
# 分阶段计时器
# ============================================================
class StageTimer:
    """记录每个训练阶段的墙钟时间，支持嵌套标记."""

    def __init__(self):
        self.records = {}       # {stage_name: elapsed_seconds}
        self._stack = []        # 当前活跃的 (name, start_time)
        self.global_start = time.perf_counter()

    def start(self, name: str):
        self._stack.append((name, time.perf_counter()))

    def stop(self, name: str = None):
        if not self._stack:
            raise RuntimeError("StageTimer: no active stage to stop")
        rec_name, t0 = self._stack.pop()
        if name is not None and rec_name != name:
            raise RuntimeError(
                f"StageTimer: expected to stop '{rec_name}', got '{name}'")
        elapsed = time.perf_counter() - t0
        self.records[rec_name] = self.records.get(rec_name, 0.0) + elapsed
        return elapsed

    def total(self):
        return time.perf_counter() - self.global_start

    def summary(self):
        total = self.total()
        lines = ["\n" + "=" * 60,
                 "  计时汇总 (wall-clock)",
                 "=" * 60]
        for name, secs in self.records.items():
            pct = secs / total * 100 if total > 0 else 0
            lines.append(f"  {name:<25s}  {secs:8.1f}s  ({pct:5.1f}%)")
        lines.append(f"  {'TOTAL':<25s}  {total:8.1f}s  (100.0%)")
        lines.append("=" * 60)
        return "\n".join(lines)


timer = StageTimer()


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================
class EMA:
    """对模型参数维护指数移动平均.

    用法:
        ema = EMA(model, decay=0.999)
        # 每个训练步后:
        ema.update()
        # 评估时:
        ema.apply_shadow()   # 将影子参数写入模型
        ... evaluate ...
        ema.restore()        # 恢复原始参数
        # 训练结束, 永久切换:
        ema.apply_shadow()   # 不再 restore
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        # 影子参数: 深拷贝当前参数
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()
        self.backup = {}

    def update(self):
        """用当前模型参数更新影子参数."""
        for name, p in self.model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    p.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        """将影子参数写入模型, 备份原始参数."""
        self.backup = {}
        for name, p in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self):
        """从备份恢复原始参数 (撤销 apply_shadow)."""
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """用于保存 checkpoint."""
        return {
            'shadow': {k: v.cpu() for k, v in self.shadow.items()},
            'decay': self.decay,
        }

    def load_state_dict(self, state):
        self.decay = state['decay']
        for k, v in state['shadow'].items():
            if k in self.shadow:
                self.shadow[k] = v.to(self.shadow[k].device)


# ============================================================
# FDM 求解器 (Fully Implicit, Thomas Algorithm)
# ============================================================
def thomas_solve(a, b, c, d, n):
    a_, b_, c_, d_ = a.copy(), b.copy(), c.copy(), d.copy()
    for i in range(1, n):
        if abs(b_[i - 1]) < 1e-15:
            continue
        m = a_[i] / b_[i - 1]
        b_[i] -= m * c_[i - 1]
        d_[i] -= m * d_[i - 1]
    x = np.zeros(n)
    x[n - 1] = d_[n - 1] / b_[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (d_[i] - c_[i] * x[i + 1]) / b_[i]
    return x


def generate_fdm_american_put(cfg):
    S_max, T, K, r, sigma = cfg.S_max, cfg.T, cfg.K, cfg.r, cfg.sigma
    Ns, Nt = cfg.fdm_S, cfg.fdm_t
    sigma2 = sigma ** 2
    dt = T / Nt

    S = np.linspace(0, S_max, Ns + 1)
    payoff = np.maximum(K - S, 0.0)
    V = payoff.copy()

    V_all = np.zeros((Nt + 1, Ns + 1))
    V_all[Nt, :] = V.copy()

    for n in range(Nt - 1, -1, -1):
        a_coef = np.zeros(Ns + 1)
        b_coef = np.zeros(Ns + 1)
        c_coef = np.zeros(Ns + 1)
        d_coef = np.zeros(Ns + 1)

        for j in range(1, Ns):
            a_coef[j] = 0.5 * dt * (r * j - sigma2 * j ** 2)
            b_coef[j] = 1 + dt * (sigma2 * j ** 2 + r)
            c_coef[j] = -0.5 * dt * (r * j + sigma2 * j ** 2)
            d_coef[j] = V[j]

        b_coef[0] = 1.0
        d_coef[0] = K
        b_coef[Ns] = 1.0
        d_coef[Ns] = 0.0

        V_new = thomas_solve(a_coef, b_coef, c_coef, d_coef, Ns + 1)
        V_new = np.maximum(V_new, payoff)
        V = V_new.copy()
        V_all[n, :] = V.copy()

    t_grid = np.linspace(0, T, Nt + 1)
    return S, t_grid, V_all


# ============================================================
# 生成 FDM 参考解
# ============================================================
timer.start("FDM")
print("生成 FDM 参考解...")
S_fdm, t_fdm, V_fdm = generate_fdm_american_put(cfg)
fdm_elapsed = timer.stop("FDM")
print(f"FDM 完成. 耗时 {fdm_elapsed:.1f}s, shape={V_fdm.shape}, "
      f"V range: [{V_fdm.min():.4f}, {V_fdm.max():.4f}]")

fdm_interp = RegularGridInterpolator(
    (t_fdm, S_fdm), V_fdm, method='linear', bounds_error=False, fill_value=None
)


# 提取 FDM 自由边界
def extract_fdm_boundary(S_fdm, t_fdm, V_fdm, K):
    payoff = np.maximum(K - S_fdm, 0.0)
    boundary = np.zeros(len(t_fdm))
    for i in range(len(t_fdm)):
        diff = V_fdm[i, :] - payoff
        in_the_money = payoff > 0
        exercise_mask = (diff < 1e-3 * K) & in_the_money
        not_exercise = ~exercise_mask & in_the_money
        if np.any(not_exercise):
            idx = np.where(not_exercise)[0]
            valid = idx[idx < np.searchsorted(S_fdm, K)]
            if len(valid) > 0:
                boundary[i] = S_fdm[valid[0]]
            else:
                boundary[i] = S_fdm[0]
        else:
            boundary[i] = S_fdm[0]
    return boundary


fdm_boundary = extract_fdm_boundary(S_fdm, t_fdm, V_fdm, cfg.K)


# ============================================================
# 在 PINN 网格上准备数据
# ============================================================
timer.start("DataPrep")

def get_fdm_on_grid(cfg, fdm_interp):
    S_np = np.linspace(0, cfg.S_max, cfg.grid_S)
    t_np = np.linspace(0, cfg.T, cfg.grid_t)
    tt_np, ss_np = np.meshgrid(t_np, S_np, indexing='ij')
    points = np.stack([tt_np.ravel(), ss_np.ravel()], axis=-1)
    V_ref = fdm_interp(points).reshape(cfg.grid_t, cfg.grid_S)
    return torch.tensor(V_ref, dtype=torch.float32).to(cfg.device)


V_fdm_grid = get_fdm_on_grid(cfg, fdm_interp)
V_fdm_norm = V_fdm_grid / cfg.K


def build_phi_target(cfg, S_fdm, t_fdm, V_fdm):
    payoff_fdm = np.maximum(cfg.K - S_fdm, 0.0)
    phi_fdm = np.ones_like(V_fdm)
    for i in range(V_fdm.shape[0]):
        diff = V_fdm[i, :] - payoff_fdm
        in_the_money = payoff_fdm > 0
        exercise_mask = (diff < 1e-3 * cfg.K) & in_the_money
        phi_fdm[i, exercise_mask] = 0.0

    phi_interp = RegularGridInterpolator(
        (t_fdm, S_fdm), phi_fdm,
        method='nearest', bounds_error=False, fill_value=1.0
    )
    S_pinn = np.linspace(0, cfg.S_max, cfg.grid_S)
    t_pinn = np.linspace(0, cfg.T, cfg.grid_t)
    tt_p, ss_p = np.meshgrid(t_pinn, S_pinn, indexing='ij')
    points = np.stack([tt_p.ravel(), ss_p.ravel()], axis=-1)
    phi_hard = phi_interp(points).reshape(cfg.grid_t, cfg.grid_S)

    continuation = (phi_hard > 0.5).astype(float)
    dist_pos = distance_transform_edt(continuation)
    dist_neg = distance_transform_edt(1 - continuation)
    signed_dist = dist_pos - dist_neg

    transition_width = 1.2
    phi_smooth = 1.0 / (1.0 + np.exp(-signed_dist / (transition_width * 0.5)))
    phi_smooth = np.clip(phi_smooth, 0.02, 0.98)

    return torch.tensor(phi_smooth, dtype=torch.float32).to(cfg.device)


phi_target = build_phi_target(cfg, S_fdm, t_fdm, V_fdm)
print(f"φ 目标标签: mean={phi_target.mean():.4f}, min={phi_target.min():.4f}, "
      f"max={phi_target.max():.4f}, std={phi_target.std():.4f}")
print(f"  行权区 (φ<0.1): {(phi_target < 0.1).float().mean():.4f}")
print(f"  继续区 (φ>0.9): {(phi_target > 0.9).float().mean():.4f}")

# 构建归一化网格坐标
S_vals = torch.linspace(0, 1, cfg.grid_S, device=cfg.device)
t_vals = torch.linspace(0, 1, cfg.grid_t, device=cfg.device)
tt, ss = torch.meshgrid(t_vals, S_vals, indexing='ij')

S_grid = ss.unsqueeze(0).unsqueeze(0)
t_grid = tt.unsqueeze(0).unsqueeze(0)

S_flat = ss.reshape(-1, 1)
t_flat = tt.reshape(-1, 1)
N_grid = S_flat.shape[0]

payoff_grid = torch.clamp(cfg.K - ss * cfg.S_max, min=0.0) / cfg.K
payoff_flat = payoff_grid.reshape(-1, 1)

exercise_ratio = (phi_target < 0.5).float().mean().item()
print(f"  FDM 行权区比例: {exercise_ratio:.4f}")

dS_norm = 1.0 / (cfg.grid_S - 1)
dt_norm = 1.0 / (cfg.grid_t - 1)

timer.stop("DataPrep")


# ============================================================
# 网络定义
# ============================================================
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, omega_0=5.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            dim = in_f
            if is_first:
                bound = 1.0 / dim
            else:
                bound = np.sqrt(6.0 / dim) / omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ValueMLP(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, omega_0=5.0):
        super().__init__()
        layers = [SirenLayer(2, hidden_dim, omega_0=omega_0, is_first=True)]
        for _ in range(num_layers - 2):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_dim) / omega_0
            self.final.weight.uniform_(-bound, bound)
            self.final.bias.zero_()

    def forward(self, S_norm, t_norm):
        x = torch.cat([S_norm, t_norm], dim=-1)
        return self.final(self.net(x))


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
    def __init__(self, in_channels=3, base_ch=32, levels=3):
        super().__init__()
        self.levels = levels
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(levels):
            out_ch = base_ch * (2 ** i)
            self.encoders.append(ConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch
        self.bottleneck = ConvBlock(ch, ch * 2)
        ch = ch * 2
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(levels - 1, -1, -1):
            out_ch = base_ch * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(ch, out_ch, 2, stride=2))
            self.decoders.append(ConvBlock(out_ch * 2, out_ch))
            ch = out_ch
        self.final = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        skips = []
        h = x
        for enc, pool in zip(self.encoders, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)
        h = self.bottleneck(h)
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = upconv(h)
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)
        return torch.sigmoid(self.final(h))


# ============================================================
# 初始化网络
# ============================================================
v_net = ValueMLP(cfg.mlp_hidden, cfg.mlp_layers, cfg.omega_0).to(cfg.device)
phi_net = PhiUNet(cfg.cnn_in_channels, cfg.cnn_base_ch, cfg.cnn_levels).to(cfg.device)
print(f"V_net 参数量: {sum(p.numel() for p in v_net.parameters()):,}")
print(f"Phi_net 参数量: {sum(p.numel() for p in phi_net.parameters()):,}")


# ============================================================
# 辅助函数
# ============================================================
def compute_bs_residual(v_net, S_norm, t_norm, cfg):
    S_n = S_norm.detach().requires_grad_(True)
    t_n = t_norm.detach().requires_grad_(True)
    V_norm = v_net(S_n, t_n)
    ones = torch.ones_like(V_norm)

    dV_dS = torch.autograd.grad(V_norm, S_n, grad_outputs=ones, create_graph=True)[0]
    dV_dt = torch.autograd.grad(V_norm, t_n, grad_outputs=ones, create_graph=True)[0]
    d2V_dS2 = torch.autograd.grad(dV_dS, S_n, grad_outputs=ones, create_graph=True)[0]

    S = S_n * cfg.S_max
    V_t = (cfg.K / cfg.T) * dV_dt
    V_S = (cfg.K / cfg.S_max) * dV_dS
    V_SS = (cfg.K / cfg.S_max ** 2) * d2V_dS2
    V = cfg.K * V_norm

    residual = V_t + 0.5 * cfg.sigma ** 2 * S ** 2 * V_SS + cfg.r * S * V_S - cfg.r * V
    return V_norm, residual / cfg.K


def phi_gradient_fd_inner(phi, dS, dt):
    if phi.dim() == 2:
        phi = phi.unsqueeze(0).unsqueeze(0)

    dphi_dS = (phi[:, :, :, 2:] - phi[:, :, :, :-2]) / (2 * dS)
    dphi_dt = (phi[:, :, 2:, :] - phi[:, :, :-2, :]) / (2 * dt)

    dphi_dS_inner = dphi_dS[:, :, 1:-1, :]
    dphi_dt_inner = dphi_dt[:, :, :, 1:-1]

    return (dphi_dS_inner ** 2 + dphi_dt_inner ** 2).squeeze()


def focal_bce(pred, target, gamma=2.0):
    pred = pred.clamp(1e-6, 1 - 1e-6)
    bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
    p_t = pred * target + (1 - pred) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def compute_bc_loss(v_net, cfg):
    n_bc = cfg.grid_t
    t_bc = torch.linspace(0, 1, n_bc, device=cfg.device).unsqueeze(1)

    bc_left = F.mse_loss(
        v_net(torch.zeros(n_bc, 1, device=cfg.device), t_bc),
        torch.ones(n_bc, 1, device=cfg.device)
    )
    bc_right = F.mse_loss(
        v_net(torch.ones(n_bc, 1, device=cfg.device), t_bc),
        torch.zeros(n_bc, 1, device=cfg.device)
    )

    n_tc = cfg.grid_S
    S_tc = torch.linspace(0, 1, n_tc, device=cfg.device).unsqueeze(1)
    t_tc = torch.ones(n_tc, 1, device=cfg.device)
    target_tc = torch.clamp(1.0 - S_tc * cfg.S_max / cfg.K, min=0.0)
    bc_terminal = F.mse_loss(v_net(S_tc, t_tc), target_tc)

    return bc_left + bc_right + bc_terminal


def compute_rel_err(v_net, S_flat, t_flat, V_fdm_grid, cfg):
    with torch.no_grad():
        V_pred = v_net(S_flat, t_flat).reshape(cfg.grid_t, cfg.grid_S) * cfg.K
        return (torch.norm(V_pred - V_fdm_grid) / (torch.norm(V_fdm_grid) + 1e-8)).item()


# ============================================================
# Stage 1: MLP 预热 (FDM 监督)
# ============================================================
print("\n" + "=" * 70)
print("Stage 1: MLP预热 (FDM监督)")
print("=" * 70)
timer.start("Stage1_MLP")

V_target_flat = V_fdm_norm.reshape(-1, 1)

opt1 = torch.optim.AdamW(v_net.parameters(), lr=cfg.stage1_lr, weight_decay=1e-5)
sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg.stage1_epochs, eta_min=1e-5)
history_s1 = {'mse': [], 'bc': [], 'rel_err': []}

for epoch in range(1, cfg.stage1_epochs + 1):
    v_net.train()

    V_pred = v_net(S_flat, t_flat)
    mse_loss = F.mse_loss(V_pred, V_target_flat)
    bc_loss = compute_bc_loss(v_net, cfg)
    loss = mse_loss + cfg.lambda_bc * bc_loss

    opt1.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
    opt1.step()
    sch1.step()

    rel_err = compute_rel_err(v_net, S_flat, t_flat, V_fdm_grid, cfg)
    history_s1['mse'].append(mse_loss.item())
    history_s1['bc'].append(bc_loss.item())
    history_s1['rel_err'].append(rel_err)

    if epoch % 500 == 0:
        vp = V_pred.detach().reshape(-1)
        print(f"  Epoch {epoch}/{cfg.stage1_epochs} | "
              f"MSE={mse_loss.item():.6f} | BC={bc_loss.item():.6f} | "
              f"RelErr={rel_err:.4f} | "
              f"V pred [mean={vp.mean():.4f}, min={vp.min():.4f}, max={vp.max():.4f}]")

s1_time = timer.stop("Stage1_MLP")
print(f"Stage 1 完成. RelErr = {history_s1['rel_err'][-1]:.6f}, 耗时 {s1_time:.1f}s")


# ============================================================
# Stage 1.5: φ Warm-Start (CNN 学习 FDM 行权区标签)
# ============================================================
print("\n" + "=" * 70)
print("Stage 1.5: φ Warm-Start")
print("=" * 70)
timer.start("Stage1.5_CNN")

phi_target_4d = phi_target.unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    V_init_map = v_net(S_flat, t_flat).reshape(1, 1, cfg.grid_t, cfg.grid_S)

cnn_input_ws = torch.cat([S_grid, t_grid, V_init_map.detach()], dim=1)

opt15 = torch.optim.Adam(phi_net.parameters(), lr=cfg.stage15_lr)
sch15 = torch.optim.lr_scheduler.CosineAnnealingLR(opt15, T_max=cfg.stage15_epochs, eta_min=1e-5)

history_s1_5 = {'stage15_bce_list': [],
                'stage15_phi_mean_list': [],
                'stage15_phi_std_list': [],
                'stage15_phi_min_list': [],
                'stage15_phi_max_list': [],
                'stage15_ex_frac_list': [],
                'stage15_cont_frac_list': [],
                }

log_interval_15 = 300

for epoch in range(1, cfg.stage15_epochs + 1):
    phi_net.train()

    phi_pred = phi_net(cnn_input_ws)
    loss = F.binary_cross_entropy(phi_pred, phi_target_4d)

    opt15.zero_grad()
    loss.backward()
    opt15.step()
    sch15.step()

    if epoch % log_interval_15 == 0:
        with torch.no_grad():
            p = phi_pred.squeeze()
            bce_val = loss.item()
            p_mean = p.mean().item()
            p_std = p.std().item()
            p_min = p.min().item()
            p_max = p.max().item()
            ex_frac = (p < 0.1).float().mean().item()
            cont_frac = (p > 0.9).float().mean().item()

            history_s1_5['stage15_bce_list'].append(bce_val)
            history_s1_5['stage15_phi_mean_list'].append(p_mean)
            history_s1_5['stage15_phi_std_list'].append(p_std)
            history_s1_5['stage15_phi_min_list'].append(p_min)
            history_s1_5['stage15_phi_max_list'].append(p_max)
            history_s1_5['stage15_ex_frac_list'].append(ex_frac)
            history_s1_5['stage15_cont_frac_list'].append(cont_frac)

            print(f"  Epoch {epoch}/{cfg.stage15_epochs} | "
                  f"BCE={bce_val:.6f} | "
                  f"φ [mean={p_mean:.4f}, min={p_min:.4f}, max={p_max:.4f}, "
                  f"std={p_std:.4f}] | "
                  f"φ<0.1: {ex_frac:.3f} | "
                  f"φ>0.9: {cont_frac:.3f}")

with torch.no_grad():
    p = phi_net(cnn_input_ws).squeeze()
    print(f"\nWarm-start 完成: mean={p.mean():.4f}, std={p.std():.4f}, "
          f"min={p.min():.4f}, max={p.max():.4f}")
    print(f"  行权区(φ<0.1): {(p < 0.1).float().mean():.1%} | "
          f"继续区(φ>0.9): {(p > 0.9).float().mean():.1%}")

s15_time = timer.stop("Stage1.5_CNN")
print(f"Stage 1.5 耗时 {s15_time:.1f}s")


# ============================================================
# Stage 2: 联合训练 (Phase-Field Energy + Anchor + EMA)
# ============================================================
print("\n" + "=" * 70)
print("Stage 2: 联合训练 (Phase-Field Energy + Anchor + EMA)")
print("=" * 70)
timer.start("Stage2_Joint")

opt_mlp = torch.optim.AdamW(v_net.parameters(), lr=cfg.stage2_lr_mlp, weight_decay=1e-5)
opt_cnn = torch.optim.AdamW(phi_net.parameters(), lr=cfg.stage2_lr_cnn, weight_decay=1e-5)
sch_mlp = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mlp, T_max=cfg.stage2_epochs, eta_min=1e-6)
sch_cnn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cnn, T_max=cfg.stage2_epochs, eta_min=1e-5)

# 初始化 EMA (从 Stage 1 结束后的 v_net 参数开始)
ema = EMA(v_net, decay=cfg.ema_decay)

history_s2 = {
    'pde': [], 'exercise': [], 'interface': [], 'anchor': [],
    'balance': [], 'bc': [], 'total': [],
    'rel_err_raw': [], 'rel_err_ema': []
}

phi_anchor_flat = phi_target.reshape(-1, 1)

for epoch in range(1, cfg.stage2_epochs + 1):
    v_net.train()
    phi_net.train()

    # ---- 1. V 前向 + BS 残差 ----
    V_norm_pred, bs_residual = compute_bs_residual(v_net, S_flat, t_flat, cfg)

    # ---- 2. φ (CNN) ----
    V_map = V_norm_pred.detach().reshape(1, 1, cfg.grid_t, cfg.grid_S)
    cnn_input = torch.cat([S_grid, t_grid, V_map], dim=1)
    phi_full = phi_net(cnn_input)
    phi_2d = phi_full.squeeze()
    phi_flat = phi_2d.reshape(-1, 1)

    # ---- 3. Phase-Field Energy ----
    pde_loss = (phi_flat * bs_residual ** 2).mean()
    exercise_loss = ((1 - phi_flat) ** 2 * (V_norm_pred - payoff_flat) ** 2).mean() / cfg.eps

    phi_inner = phi_2d[1:-1, 1:-1]
    W_phi_inner = phi_inner ** 2 * (1 - phi_inner) ** 2
    grad_phi_sq = phi_gradient_fd_inner(phi_2d, dS_norm, dt_norm)
    interface_loss = (cfg.eps * grad_phi_sq + W_phi_inner / cfg.eps).mean()

    # ---- 4. Anchor (Focal BCE) ----
    anchor_loss = focal_bce(phi_flat, phi_anchor_flat, gamma=2.0)

    # ---- 5. Balance ----
    target_phi_mean = 1.0 - exercise_ratio
    balance_loss = (phi_2d.mean() - target_phi_mean) ** 2

    # ---- 6. BC ----
    bc_loss = compute_bc_loss(v_net, cfg)

    # ---- Anchor 权重衰减 ----
    if epoch < cfg.anchor_decay_start:
        anchor_w = cfg.lambda_anchor
    elif epoch >= cfg.anchor_decay_end:
        anchor_w = cfg.lambda_anchor * cfg.anchor_min_ratio
    else:
        progress = (epoch - cfg.anchor_decay_start) / (cfg.anchor_decay_end - cfg.anchor_decay_start)
        anchor_w = cfg.lambda_anchor * (1.0 - progress * (1.0 - cfg.anchor_min_ratio))

    # ---- 总损失 ----
    total = (cfg.lambda_pde * pde_loss
             + cfg.lambda_ex * exercise_loss
             + cfg.lambda_int * interface_loss
             + anchor_w * anchor_loss
             + cfg.lambda_balance * balance_loss
             + cfg.lambda_bc * bc_loss)

    # ---- 反向传播 ----
    opt_mlp.zero_grad()
    opt_cnn.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(phi_net.parameters(), 1.0)
    opt_mlp.step()
    opt_cnn.step()
    sch_mlp.step()
    sch_cnn.step()

    # ---- EMA 更新 (每步都更新) ----
    ema.update()

    # ---- 记录 ----
    rel_err_raw = compute_rel_err(v_net, S_flat, t_flat, V_fdm_grid, cfg)

    # EMA RelErr: 临时切换到影子参数评估
    ema.apply_shadow()
    rel_err_ema = compute_rel_err(v_net, S_flat, t_flat, V_fdm_grid, cfg)
    ema.restore()

    history_s2['pde'].append(pde_loss.item())
    history_s2['exercise'].append(exercise_loss.item())
    history_s2['interface'].append(interface_loss.item())
    history_s2['anchor'].append(anchor_loss.item())
    history_s2['balance'].append(balance_loss.item())
    history_s2['bc'].append(bc_loss.item())
    history_s2['total'].append(total.item())
    history_s2['rel_err_raw'].append(rel_err_raw)
    history_s2['rel_err_ema'].append(rel_err_ema)

    if epoch % 1000 == 0:
        with torch.no_grad():
            p = phi_2d
            print(f"  Epoch {epoch}/{cfg.stage2_epochs} | "
                  f"PDE={pde_loss.item():.6f} | Ex={exercise_loss.item():.6f} | "
                  f"Int={interface_loss.item():.4f} | Anc={anchor_loss.item():.6f} | "
                  f"Bal={balance_loss.item():.6f} | BC={bc_loss.item():.6f} | "
                  f"Total={total.item():.4f} | "
                  f"RelErr(raw)={rel_err_raw:.4f} | RelErr(ema)={rel_err_ema:.4f} | "
                  f"AncW={anchor_w:.1f} | "
                  f"φ [mean={p.mean():.3f}, std={p.std():.3f}]")

s2_time = timer.stop("Stage2_Joint")
print(f"\nStage 2 完成. 耗时 {s2_time:.1f}s")
print(f"  最终 RelErr(raw) = {history_s2['rel_err_raw'][-1]:.6f}")
print(f"  最终 RelErr(ema) = {history_s2['rel_err_ema'][-1]:.6f}")


# ============================================================
# Stage 2 → Stage 3 过渡: 保存 raw 参数, 切换到 EMA
# ============================================================
# 保存 raw (非 EMA) 的 checkpoint
torch.save({
    'v_net_raw': v_net.state_dict(),
    'phi_net': phi_net.state_dict(),
    'ema': ema.state_dict(),
}, 'scheme1_stage2_raw.pth')
print("已保存 Stage 2 raw checkpoint → scheme1_stage2_raw.pth")

# 永久切换到 EMA 参数 (不再 restore)
ema.apply_shadow()
rel_err_after_ema = compute_rel_err(v_net, S_flat, t_flat, V_fdm_grid, cfg)
print(f"EMA 参数已应用. RelErr(ema→active) = {rel_err_after_ema:.6f}")


# ============================================================
# Stage 3: L-BFGS 精调 (固定 φ, 从 EMA 参数出发)
# ============================================================
print("\n" + "=" * 70)
print("Stage 3: L-BFGS 精调 (从 EMA 参数出发)")
print("=" * 70)
timer.start("Stage3_LBFGS")

phi_net.eval()
for p_param in phi_net.parameters():
    p_param.requires_grad_(False)

with torch.no_grad():
    V_map_s3 = v_net(S_flat, t_flat).reshape(1, 1, cfg.grid_t, cfg.grid_S)
    cnn_in_s3 = torch.cat([S_grid, t_grid, V_map_s3], dim=1)
    phi_fixed = phi_net(cnn_in_s3).squeeze()
    phi_fixed_flat = phi_fixed.reshape(-1, 1)

lbfgs = torch.optim.LBFGS(
    v_net.parameters(), lr=0.5, max_iter=5,
    history_size=20, line_search_fn='strong_wolfe'
)
history_s3 = {'total': [], 'rel_err': []}

for step in range(cfg.stage3_steps):
    def closure():
        lbfgs.zero_grad()
        V_n, bs_res = compute_bs_residual(v_net, S_flat, t_flat, cfg)

        pde = (phi_fixed_flat * bs_res ** 2).mean()
        ex = ((1 - phi_fixed_flat) ** 2 * (V_n - payoff_flat) ** 2).mean() / cfg.eps
        bc = compute_bc_loss(v_net, cfg)

        loss = cfg.lambda_pde * pde + cfg.lambda_ex * ex + cfg.lambda_bc * bc
        loss.backward()
        return loss

    loss_val = lbfgs.step(closure)

    rel_err = compute_rel_err(v_net, S_flat, t_flat, V_fdm_grid, cfg)
    lv = loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val
    history_s3['total'].append(lv)
    history_s3['rel_err'].append(rel_err)

    if (step + 1) % 10 == 0:
        print(f"  Step {step + 1}/{cfg.stage3_steps} | Loss={lv:.6f} | RelErr={rel_err:.4f}")

s3_time = timer.stop("Stage3_LBFGS")
print(f"Stage 3 完成. 耗时 {s3_time:.1f}s")


# ============================================================
# 保存结果
# ============================================================
print("\n保存结果...")
timer.start("SaveResults")

with torch.no_grad():
    V_final_norm = v_net(S_flat, t_flat).reshape(cfg.grid_t, cfg.grid_S)
    V_final = (V_final_norm * cfg.K).cpu().numpy()

    V_map_save = V_final_norm.reshape(1, 1, cfg.grid_t, cfg.grid_S)
    cnn_in_save = torch.cat([S_grid, t_grid, V_map_save], dim=1)
    phi_final = phi_net(cnn_in_save).squeeze().cpu().numpy()

S_save = np.linspace(0, cfg.S_max, cfg.grid_S)
t_save = np.linspace(0, cfg.T, cfg.grid_t)

np.savez('scheme1_results.npz',
         S=S_save, t=t_save,
         V_pred=V_final,
         V_fdm=V_fdm_grid.cpu().numpy(),
         phi=phi_final,
         phi_target=phi_target.cpu().numpy(),
         fdm_boundary=fdm_boundary,
         S_fdm=S_fdm, t_fdm=t_fdm)

history = {
    'stage1': {k: [float(v) for v in vals] for k, vals in history_s1.items()},
    'stage1_5': {k: [float(v) for v in vals] for k, vals in history_s1_5.items()},
    'stage2': {k: [float(v) for v in vals] for k, vals in history_s2.items()},
    'stage3': {k: [float(v) for v in vals] for k, vals in history_s3.items()},
}
with open('scheme1_history.json', 'w') as f:
    json.dump(history, f)

torch.save({
    'v_net': v_net.state_dict(),
    'phi_net': phi_net.state_dict(),
    'ema': ema.state_dict(),
    'config': {k: v for k, v in vars(cfg).items()
               if not k.startswith('_') and not isinstance(v, torch.device)},
}, 'scheme1_best.pth')

timer.stop("SaveResults")

# ============================================================
# 最终汇总
# ============================================================
print("\n所有结果已保存!")
print(f"  scheme1_results.npz")
print(f"  scheme1_history.json")
print(f"  scheme1_best.pth")
print(f"  scheme1_stage2_raw.pth  (非EMA参数备份)")
print(f"\n最终 V RelErr = {history_s3['rel_err'][-1]:.6f}")

# 打印计时汇总
print(timer.summary())