"""
robustness_test.py
鲁棒性验证：多参数组合自动批量实验

将本文件放在 s1_cnn_phasefield.py 同目录下运行:
    python robustness_test.py

预计总时间: ~6小时 (4组 × ~90min/组, 单核 i7-13700H)
"""

import numpy as np
import json
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

# ============================================================
# 复用 s1_cnn_phasefield.py 中的核心组件
# ============================================================
from s1_cnn_phasefield import (
    thomas_solve,
    generate_fdm_american_put,
    extract_fdm_boundary,
    SirenLayer,
    ValueMLP,
    ConvBlock,
    PhiUNet,
    compute_bs_residual,
    phi_gradient_fd_inner,
    focal_bce,
    set_seed,
)


# ============================================================
# 实验参数组
# ============================================================
EXPERIMENTS = {
    "base": {
        "sigma": 0.2, "r": 0.05, "T": 1.0, "K": 100.0,
        "S_max": 200.0, "eps": 0.05,
        "description": "Baseline (same as paper)"
    },
    "low_vol": {
        "sigma": 0.1, "r": 0.05, "T": 1.0, "K": 100.0,
        "S_max": 200.0, "eps": 0.05,
        "description": "Low volatility (sigma=0.1)"
    },
    "high_vol": {
        "sigma": 0.4, "r": 0.05, "T": 1.0, "K": 100.0,
        "S_max": 200.0, "eps": 0.05,
        "description": "High volatility (sigma=0.4)"
    },
    "low_rate": {
        "sigma": 0.2, "r": 0.02, "T": 1.0, "K": 100.0,
        "S_max": 200.0, "eps": 0.05,
        "description": "Low interest rate (r=0.02)"
    },
    "short_mat": {
        "sigma": 0.2, "r": 0.05, "T": 0.25, "K": 100.0,
        "S_max": 200.0, "eps": 0.05,
        "description": "Short maturity (T=0.25)"
    },
}


# ============================================================
# Config 类 (可按实验参数覆盖)
# ============================================================
class RobustConfig:
    def __init__(self, **kwargs):
        # 期权参数 (可覆盖)
        self.S_max = kwargs.get("S_max", 200.0)
        self.T = kwargs.get("T", 1.0)
        self.K = kwargs.get("K", 100.0)
        self.r = kwargs.get("r", 0.05)
        self.sigma = kwargs.get("sigma", 0.2)

        # Phase-Field
        self.eps = kwargs.get("eps", 0.05)
        self.lambda_int = 0.01
        self.lambda_ex = 3.0
        self.lambda_pde = 1.0
        self.lambda_bc = 50.0
        self.lambda_anchor = 25.0
        self.lambda_balance = 2.0

        # 网格
        self.grid_S = 128
        self.grid_t = 64

        # MLP
        self.mlp_hidden = 256
        self.mlp_layers = 4
        self.omega_0 = 5.0

        # CNN
        self.cnn_base_ch = 32
        self.cnn_levels = 3
        self.cnn_in_channels = 3

        # FDM
        self.fdm_S = 500
        self.fdm_t = 2000

        # 训练
        self.stage1_epochs = 3000
        self.stage1_lr = 2e-3
        self.stage15_epochs = 1500
        self.stage15_lr = 2e-3
        self.stage2_epochs = 20000
        self.stage2_lr_mlp = 5e-4
        self.stage2_lr_cnn = 5e-3
        self.stage3_steps = 50

        # Anchor 衰减
        self.anchor_decay_start = 10000
        self.anchor_decay_end = 20000
        self.anchor_min_ratio = 0.3

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 42


# ============================================================
# 辅助函数
# ============================================================
def get_fdm_on_grid(cfg, fdm_interp):
    S_np = np.linspace(0, cfg.S_max, cfg.grid_S)
    t_np = np.linspace(0, cfg.T, cfg.grid_t)
    tt_np, ss_np = np.meshgrid(t_np, S_np, indexing='ij')
    points = np.stack([tt_np.ravel(), ss_np.ravel()], axis=-1)
    V_ref = fdm_interp(points).reshape(cfg.grid_t, cfg.grid_S)
    return torch.tensor(V_ref, dtype=torch.float32).to(cfg.device)


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
# 单组实验的完整训练流程
# ============================================================
def run_single_experiment(name, params):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name} — {params['description']}")
    print("=" * 70)

    total_start = time.time()

    # --- Config ---
    cfg = RobustConfig(**params)
    set_seed(cfg.seed)

    # --- Stage 时间记录 ---
    stage_times = {}

    # --- FDM ---
    t0 = time.time()
    S_fdm, t_fdm, V_fdm = generate_fdm_american_put(cfg)
    stage_times['fdm'] = time.time() - t0
    print(f"  FDM: {stage_times['fdm']:.1f}s, shape={V_fdm.shape}, "
          f"V range: [{V_fdm.min():.4f}, {V_fdm.max():.4f}]")

    fdm_interp = RegularGridInterpolator(
        (t_fdm, S_fdm), V_fdm, method='linear',
        bounds_error=False, fill_value=None
    )
    fdm_boundary = extract_fdm_boundary(S_fdm, t_fdm, V_fdm, cfg.K)

    V_fdm_grid = get_fdm_on_grid(cfg, fdm_interp)
    V_fdm_norm = V_fdm_grid / cfg.K
    phi_target = build_phi_target(cfg, S_fdm, t_fdm, V_fdm)

    print(f"  φ target: mean={phi_target.mean():.4f}, std={phi_target.std():.4f}")
    print(f"  Exercise(φ<0.1): {(phi_target < 0.1).float().mean():.3f}, "
          f"Continuation(φ>0.9): {(phi_target > 0.9).float().mean():.3f}")

    # --- 网格坐标 ---
    S_vals = torch.linspace(0, 1, cfg.grid_S, device=cfg.device)
    t_vals = torch.linspace(0, 1, cfg.grid_t, device=cfg.device)
    tt, ss = torch.meshgrid(t_vals, S_vals, indexing='ij')

    S_grid = ss.unsqueeze(0).unsqueeze(0)
    t_grid = tt.unsqueeze(0).unsqueeze(0)
    S_flat = ss.reshape(-1, 1)
    t_flat = tt.reshape(-1, 1)

    payoff_grid = torch.clamp(cfg.K - ss * cfg.S_max, min=0.0) / cfg.K
    payoff_flat = payoff_grid.reshape(-1, 1)

    exercise_ratio = (phi_target < 0.5).float().mean().item()
    dS_norm = 1.0 / (cfg.grid_S - 1)
    dt_norm = 1.0 / (cfg.grid_t - 1)

    # --- 网络 ---
    v_net = ValueMLP(cfg.mlp_hidden, cfg.mlp_layers, cfg.omega_0).to(cfg.device)
    phi_net = PhiUNet(cfg.cnn_in_channels, cfg.cnn_base_ch, cfg.cnn_levels).to(cfg.device)

    # ===================== Stage 1 =====================
    print(f"\n  Stage 1: MLP Pre-training ({cfg.stage1_epochs} epochs)")
    t0 = time.time()

    V_target_flat = V_fdm_norm.reshape(-1, 1)
    opt1 = torch.optim.AdamW(v_net.parameters(), lr=cfg.stage1_lr, weight_decay=1e-5)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg.stage1_epochs, eta_min=1e-5)

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

        if epoch % 1000 == 0:
            rel_err = compute_rel_err(v_net, S_flat, t_flat, V_fdm_grid, cfg)
            print(f"    Epoch {epoch}: MSE={mse_loss.item():.6f}, RelErr={rel_err:.4f}")

    stage_times['stage1'] = time.time() - t0
    s1_relerr = compute_rel_err(v_net, S_flat, t_flat, V_fdm_grid, cfg)
    print(f"  Stage 1 done: {stage_times['stage1']:.1f}s, RelErr={s1_relerr:.6f}")

    # ===================== Stage 1.5 =====================
    print(f"\n  Stage 1.5: CNN Warm-Start ({cfg.stage15_epochs} epochs)")
    t0 = time.time()

    phi_target_4d = phi_target.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        V_init_map = v_net(S_flat, t_flat).reshape(1, 1, cfg.grid_t, cfg.grid_S)
    cnn_input_ws = torch.cat([S_grid, t_grid, V_init_map.detach()], dim=1)

    opt15 = torch.optim.Adam(phi_net.parameters(), lr=cfg.stage15_lr)
    sch15 = torch.optim.lr_scheduler.CosineAnnealingLR(opt15, T_max=cfg.stage15_epochs, eta_min=1e-5)

    for epoch in range(1, cfg.stage15_epochs + 1):
        phi_net.train()
        phi_pred = phi_net(cnn_input_ws)
        loss = F.binary_cross_entropy(phi_pred, phi_target_4d)
        opt15.zero_grad()
        loss.backward()
        opt15.step()
        sch15.step()

        if epoch % 500 == 0:
            p = phi_pred.squeeze()
            print(f"    Epoch {epoch}: BCE={loss.item():.6f}, "
                  f"φ mean={p.mean().item():.4f}, std={p.std().item():.4f}")

    stage_times['stage15'] = time.time() - t0
    print(f"  Stage 1.5 done: {stage_times['stage15']:.1f}s")

    # ===================== Stage 2 =====================
    print(f"\n  Stage 2: Joint Training ({cfg.stage2_epochs} epochs)")
    t0 = time.time()

    opt_mlp = torch.optim.AdamW(v_net.parameters(), lr=cfg.stage2_lr_mlp, weight_decay=1e-5)
    opt_cnn = torch.optim.AdamW(phi_net.parameters(), lr=cfg.stage2_lr_cnn, weight_decay=1e-5)
    sch_mlp = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mlp, T_max=cfg.stage2_epochs, eta_min=1e-6)
    sch_cnn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cnn, T_max=cfg.stage2_epochs, eta_min=1e-5)

    phi_anchor_flat = phi_target.reshape(-1, 1)

    for epoch in range(1, cfg.stage2_epochs + 1):
        v_net.train()
        phi_net.train()

        S_n = S_flat.detach().requires_grad_(True)
        t_n = t_flat.detach().requires_grad_(True)
        V_norm_pred, bs_residual = compute_bs_residual(v_net, S_n, t_n, cfg)

        V_map = V_norm_pred.detach().reshape(1, 1, cfg.grid_t, cfg.grid_S)
        cnn_input = torch.cat([S_grid, t_grid, V_map], dim=1)
        phi_full = phi_net(cnn_input)
        phi_2d = phi_full.squeeze()
        phi_flat = phi_2d.reshape(-1, 1)

        pde_loss = (phi_flat * bs_residual ** 2).mean()
        exercise_loss = ((1 - phi_flat) ** 2 * (V_norm_pred - payoff_flat) ** 2).mean() / cfg.eps

        phi_inner = phi_2d[1:-1, 1:-1]
        W_phi_inner = phi_inner ** 2 * (1 - phi_inner) ** 2
        grad_phi_sq = phi_gradient_fd_inner(phi_2d, dS_norm, dt_norm)
        interface_loss = (cfg.eps * grad_phi_sq + W_phi_inner / cfg.eps).mean()

        anchor_loss = focal_bce(phi_flat, phi_anchor_flat, gamma=2.0)

        target_phi_mean = 1.0 - exercise_ratio
        balance_loss = (phi_2d.mean() - target_phi_mean) ** 2

        bc_loss = compute_bc_loss(v_net, cfg)

        # Anchor 衰减
        if epoch < cfg.anchor_decay_start:
            anchor_w = cfg.lambda_anchor
        elif epoch >= cfg.anchor_decay_end:
            anchor_w = cfg.lambda_anchor * cfg.anchor_min_ratio
        else:
            progress = (epoch - cfg.anchor_decay_start) / (cfg.anchor_decay_end - cfg.anchor_decay_start)
            anchor_w = cfg.lambda_anchor * (1.0 - progress * (1.0 - cfg.anchor_min_ratio))

        total = (cfg.lambda_pde * pde_loss
                 + cfg.lambda_ex * exercise_loss
                 + cfg.lambda_int * interface_loss
                 + anchor_w * anchor_loss
                 + cfg.lambda_balance * balance_loss
                 + cfg.lambda_bc * bc_loss)

        opt_mlp.zero_grad()
        opt_cnn.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(phi_net.parameters(), 1.0)
        opt_mlp.step()
        opt_cnn.step()
        sch_mlp.step()
        sch_cnn.step()

        if epoch % 5000 == 0:
            rel_err = compute_rel_err(v_net, S_flat, t_flat, V_fdm_grid, cfg)
            print(f"    Epoch {epoch}: Total={total.item():.4f}, RelErr={rel_err:.4f}, "
                  f"φ mean={phi_2d.mean().item():.3f}, std={phi_2d.std().item():.3f}")

    stage_times['stage2'] = time.time() - t0
    s2_relerr = compute_rel_err(v_net, S_flat, t_flat, V_fdm_grid, cfg)
    print(f"  Stage 2 done: {stage_times['stage2']:.1f}s, RelErr={s2_relerr:.6f}")

    # ===================== Stage 3 =====================
    print(f"\n  Stage 3: L-BFGS ({cfg.stage3_steps} steps)")
    t0 = time.time()

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

    for step in range(cfg.stage3_steps):
        def closure():
            lbfgs.zero_grad()
            S_n = S_flat.detach().requires_grad_(True)
            t_n = t_flat.detach().requires_grad_(True)
            V_n, bs_res = compute_bs_residual(v_net, S_n, t_n, cfg)
            pde = (phi_fixed_flat * bs_res ** 2).mean()
            ex = ((1 - phi_fixed_flat) ** 2 * (V_n - payoff_flat) ** 2).mean() / cfg.eps
            bc = compute_bc_loss(v_net, cfg)
            loss = cfg.lambda_pde * pde + cfg.lambda_ex * ex + cfg.lambda_bc * bc
            loss.backward()
            return loss
        lbfgs.step(closure)

    stage_times['stage3'] = time.time() - t0
    final_relerr = compute_rel_err(v_net, S_flat, t_flat, V_fdm_grid, cfg)
    print(f"  Stage 3 done: {stage_times['stage3']:.1f}s, RelErr={final_relerr:.6f}")

    # ===================== 收集结果 =====================
    with torch.no_grad():
        V_final_norm = v_net(S_flat, t_flat).reshape(cfg.grid_t, cfg.grid_S)
        V_final = (V_final_norm * cfg.K).cpu().numpy()

        V_map_save = V_final_norm.reshape(1, 1, cfg.grid_t, cfg.grid_S)
        cnn_in_save = torch.cat([S_grid, t_grid, V_map_save], dim=1)

        # 恢复 phi_net 的 eval 模式下的前向
        phi_final = phi_net(cnn_in_save).squeeze().cpu().numpy()

    phi_mean = float(np.mean(phi_final))
    phi_std = float(np.std(phi_final))
    phi_min = float(np.min(phi_final))
    phi_max = float(np.max(phi_final))
    ex_frac = float(np.mean(phi_final < 0.1))
    cont_frac = float(np.mean(phi_final > 0.9))

    # ATM value
    S_save = np.linspace(0, cfg.S_max, cfg.grid_S)
    idx_K = np.argmin(np.abs(S_save - cfg.K))
    V_atm_pred = V_final[0, idx_K]
    V_atm_fdm = V_fdm_grid.cpu().numpy()[0, idx_K]

    # 自由边界误差
    phi_torch = torch.tensor(phi_final, dtype=torch.float32)
    t_save = np.linspace(0, cfg.T, cfg.grid_t)
    fb_pred = np.full(cfg.grid_t, np.nan)
    for i in range(cfg.grid_t):
        row = phi_final[i, :]
        crossings = np.where(np.diff(np.sign(row - 0.5)))[0]
        if len(crossings) > 0:
            j = crossings[0]
            if abs(row[j+1] - row[j]) > 1e-10:
                alpha = (0.5 - row[j]) / (row[j+1] - row[j])
                fb_pred[i] = S_save[j] + alpha * (S_save[j+1] - S_save[j])
            else:
                fb_pred[i] = S_save[j]

    # FDM 边界插值到 PINN 时间网格
    from scipy.interpolate import interp1d
    fb_fdm_interp = interp1d(t_fdm, fdm_boundary, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
    fb_fdm_on_pinn = fb_fdm_interp(t_save)

    valid = ~np.isnan(fb_pred) & (t_save > 0.02 * cfg.T) & (t_save < 0.98 * cfg.T)
    if np.any(valid):
        fb_mean_err = float(np.mean(np.abs(fb_pred[valid] - fb_fdm_on_pinn[valid])))
        fb_max_err = float(np.max(np.abs(fb_pred[valid] - fb_fdm_on_pinn[valid])))
        fb_rel_err = float(fb_mean_err / (np.mean(np.abs(fb_fdm_on_pinn[valid])) + 1e-8))
    else:
        fb_mean_err = fb_max_err = fb_rel_err = float('nan')

    total_time = time.time() - total_start

    result = {
        "name": name,
        "description": params["description"],
        "params": {k: v for k, v in params.items() if k != "description"},
        "final_relerr": final_relerr,
        "stage1_relerr": s1_relerr,
        "stage2_relerr": s2_relerr,
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "phi_min": phi_min,
        "phi_max": phi_max,
        "ex_frac": ex_frac,
        "cont_frac": cont_frac,
        "V_atm_pred": float(V_atm_pred),
        "V_atm_fdm": float(V_atm_fdm),
        "fb_mean_err": fb_mean_err,
        "fb_max_err": fb_max_err,
        "fb_rel_err": fb_rel_err,
        "stage_times": stage_times,
        "total_time": total_time,
    }

    print(f"\n  === {name} SUMMARY ===")
    print(f"  RelErr = {final_relerr:.6f} ({final_relerr*100:.4f}%)")
    print(f"  φ: mean={phi_mean:.4f}, std={phi_std:.4f}")
    print(f"  FB: mean|err|={fb_mean_err:.2f}, rel={fb_rel_err:.4f}")
    print(f"  V_ATM: pred={V_atm_pred:.4f}, fdm={V_atm_fdm:.4f}")
    print(f"  Total time: {total_time:.1f}s")

    return result


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("ROBUSTNESS TEST: Phase-Field Dual-Network American Put")
    print("=" * 70)

    results = {}
    for name, params in EXPERIMENTS.items():
        results[name] = run_single_experiment(name, params)

    # ===================== 汇总表格 =====================
    print("\n\n" + "=" * 100)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 100)
    print(f"\n{'Case':<12} {'σ':<6} {'r':<6} {'T':<6} "
          f"{'RelErr':<12} {'φ mean':<10} {'φ std':<10} "
          f"{'FB mean|e|':<12} {'FB rel':<10} {'Time(s)':<10}")
    print("-" * 100)

    for name, res in results.items():
        p = res["params"]
        print(f"{name:<12} {p['sigma']:<6} {p['r']:<6} {p['T']:<6} "
              f"{res['final_relerr']:<12.6f} "
              f"{res['phi_mean']:<10.4f} {res['phi_std']:<10.4f} "
              f"{res['fb_mean_err']:<12.2f} {res['fb_rel_err']:<10.4f} "
              f"{res['total_time']:<10.1f}")

    # ===================== LaTeX 表格 =====================
    print("\n\n% === LaTeX table (copy to paper) ===")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Robustness study across different option parameters.}")
    print(r"\label{tab:robustness}")
    print(r"\begin{tabular}{@{}lccccccr@{}}")
    print(r"\toprule")
    print(r"Case & $\sigma$ & $r$ & $T$ & RelErr & "
          r"$\bar{\varphi}$ & FB rel.\ err. & Time (s) \\")
    print(r"\midrule")
    for name, res in results.items():
        p = res["params"]
        relerr_pct = f"{res['final_relerr']*100:.2f}\\%"
        fb_pct = f"{res['fb_rel_err']*100:.2f}\\%"
        case_label = name.replace("_", " ").title()
        print(f"{case_label} & ${p['sigma']}$ & ${p['r']}$ & ${p['T']}$ & "
              f"${relerr_pct}$ & ${res['phi_mean']:.3f}$ & "
              f"${fb_pct}$ & ${res['total_time']:.0f}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # ===================== 保存 =====================
    output_file = "robustness_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print("\nDone! Total experiments:", len(results))


if __name__ == "__main__":
    main()