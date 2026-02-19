#!/usr/bin/env python3
"""
appendix_experiments.py
============================
Self-contained script — does NOT import s1_cnn_phasefield.py
(to avoid triggering its module-level training code).

Part 1: Re-run Stage 2 with φ snapshots (baseline, ~80 min)
Part 2: ε sensitivity study (4 new cases, ~5-6 hr)

Outputs (appendix_data/):
  phi_snapshots.npz, phi_snapshots_meta.json
  epsilon_sensitivity.json, epsilon_sensitivity.npz

"""

import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

OUT_DIR = 'appendix_data'
os.makedirs(OUT_DIR, exist_ok=True)


# ==================================================================
# Config (copied from source)
# ==================================================================
class Config:
    S_max = 200.0; T = 1.0; K = 100.0; r = 0.05; sigma = 0.2
    eps = 0.05
    lambda_int = 0.01; lambda_ex = 3.0; lambda_pde = 1.0
    lambda_bc = 50.0; lambda_anchor = 25.0; lambda_balance = 2.0
    grid_S = 128; grid_t = 64
    mlp_hidden = 256; mlp_layers = 4; omega_0 = 5.0
    cnn_base_ch = 32; cnn_levels = 3; cnn_in_channels = 3
    fdm_S = 500; fdm_t = 2000
    stage1_epochs = 3000; stage1_lr = 2e-3
    stage15_epochs = 1500; stage15_lr = 2e-3
    stage2_epochs = 20000; stage2_lr_mlp = 5e-4; stage2_lr_cnn = 5e-3
    stage3_steps = 50
    anchor_decay_start = 10000; anchor_decay_end = 20000
    anchor_min_ratio = 0.3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42


# ==================================================================
# Network definitions (copied from source)
# ==================================================================
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
            nn.GroupNorm(min(8, out_ch), out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch), nn.GELU(),
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
            h = enc(h); skips.append(h); h = pool(h)
        h = self.bottleneck(h)
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = upconv(h)
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:],
                                  mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1); h = dec(h)
        return torch.sigmoid(self.final(h))


# ==================================================================
# Helper functions (copied from source)
# ==================================================================
def thomas_solve(a, b, c, d, n):
    a_, b_, c_, d_ = a.copy(), b.copy(), c.copy(), d.copy()
    for i in range(1, n):
        if abs(b_[i-1]) < 1e-15: continue
        m = a_[i] / b_[i-1]
        b_[i] -= m * c_[i-1]; d_[i] -= m * d_[i-1]
    x = np.zeros(n)
    x[n-1] = d_[n-1] / b_[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (d_[i] - c_[i] * x[i+1]) / b_[i]
    return x


def generate_fdm(cfg):
    S_max, T, K, r, sigma = cfg.S_max, cfg.T, cfg.K, cfg.r, cfg.sigma
    Ns, Nt = cfg.fdm_S, cfg.fdm_t
    sigma2 = sigma**2; dt = T / Nt
    S = np.linspace(0, S_max, Ns+1)
    payoff = np.maximum(K - S, 0.0)
    V = payoff.copy()
    V_all = np.zeros((Nt+1, Ns+1)); V_all[Nt,:] = V.copy()
    for n in range(Nt-1, -1, -1):
        a=np.zeros(Ns+1); b=np.zeros(Ns+1)
        c_a=np.zeros(Ns+1); d=np.zeros(Ns+1)
        for j in range(1, Ns):
            a[j]=0.5*dt*(r*j - sigma2*j**2)
            b[j]=1+dt*(sigma2*j**2 + r)
            c_a[j]=-0.5*dt*(r*j + sigma2*j**2)
            d[j]=V[j]
        b[0]=1.0; d[0]=K; b[Ns]=1.0; d[Ns]=0.0
        V_new = thomas_solve(a, b, c_a, d, Ns+1)
        V_new = np.maximum(V_new, payoff)
        V = V_new.copy(); V_all[n,:] = V.copy()
    t_grid = np.linspace(0, T, Nt+1)
    return S, t_grid, V_all


def extract_boundary(S_fdm, t_fdm, V_fdm, K):
    payoff = np.maximum(K - S_fdm, 0.0)
    boundary = np.zeros(len(t_fdm))
    for i in range(len(t_fdm)):
        diff = V_fdm[i,:] - payoff
        in_money = payoff > 0
        ex_mask = (diff < 1e-3*K) & in_money
        not_ex = ~ex_mask & in_money
        if np.any(not_ex):
            idx = np.where(not_ex)[0]
            valid = idx[idx < np.searchsorted(S_fdm, K)]
            boundary[i] = S_fdm[valid[0]] if len(valid) > 0 else S_fdm[0]
        else:
            boundary[i] = S_fdm[0]
    return boundary


def build_phi_target(cfg, S_fdm, t_fdm, V_fdm, S_pinn, t_pinn):
    payoff_fdm = np.maximum(cfg.K - S_fdm, 0.0)
    phi_fdm = np.ones_like(V_fdm)
    for i in range(V_fdm.shape[0]):
        diff = V_fdm[i,:] - payoff_fdm
        in_money = payoff_fdm > 0
        exercise = (diff < 1e-3*cfg.K) & in_money
        phi_fdm[i, exercise] = 0.0
    phi_interp = RegularGridInterpolator(
        (t_fdm, S_fdm), phi_fdm, method='nearest',
        bounds_error=False, fill_value=1.0)
    tt_p, ss_p = np.meshgrid(t_pinn, S_pinn, indexing='ij')
    pts = np.stack([tt_p.ravel(), ss_p.ravel()], axis=-1)
    phi_hard = phi_interp(pts).reshape(len(t_pinn), len(S_pinn))
    cont = (phi_hard > 0.5).astype(float)
    dp = distance_transform_edt(cont)
    dn = distance_transform_edt(1 - cont)
    sd = dp - dn; w = 1.2
    phi_smooth = 1.0 / (1.0 + np.exp(-sd / (w*0.5)))
    return np.clip(phi_smooth, 0.02, 0.98)


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
    V_SS = (cfg.K / cfg.S_max**2) * d2V_dS2
    V = cfg.K * V_norm
    residual = V_t + 0.5*cfg.sigma**2*S**2*V_SS + cfg.r*S*V_S - cfg.r*V
    return V_norm, residual / cfg.K


def compute_bc_loss(v_net, cfg):
    n_bc = cfg.grid_t
    t_bc = torch.linspace(0, 1, n_bc, device=cfg.device).unsqueeze(1)
    bc_left = F.mse_loss(
        v_net(torch.zeros(n_bc,1,device=cfg.device), t_bc),
        torch.ones(n_bc,1,device=cfg.device))
    bc_right = F.mse_loss(
        v_net(torch.ones(n_bc,1,device=cfg.device), t_bc),
        torch.zeros(n_bc,1,device=cfg.device))
    n_tc = cfg.grid_S
    S_tc = torch.linspace(0, 1, n_tc, device=cfg.device).unsqueeze(1)
    t_tc = torch.ones(n_tc, 1, device=cfg.device)
    target_tc = torch.clamp(1.0 - S_tc*cfg.S_max/cfg.K, min=0.0)
    bc_term = F.mse_loss(v_net(S_tc, t_tc), target_tc)
    return bc_left + bc_right + bc_term


def compute_rel_err(v_net, S_flat, t_flat, V_fdm_grid, cfg):
    with torch.no_grad():
        V_pred = v_net(S_flat, t_flat).reshape(cfg.grid_t, cfg.grid_S) * cfg.K
        return (torch.norm(V_pred - V_fdm_grid) / (torch.norm(V_fdm_grid)+1e-8)).item()


def focal_bce(pred, target, gamma=2.0):
    pred = pred.clamp(1e-6, 1-1e-6)
    bce = -target*torch.log(pred) - (1-target)*torch.log(1-pred)
    p_t = pred*target + (1-pred)*(1-target)
    return ((1-p_t)**gamma * bce).mean()


def phi_gradient_fd_inner(phi, dS, dt):
    if phi.dim() == 2:
        phi = phi.unsqueeze(0).unsqueeze(0)
    dphi_dS = (phi[:,:,:,2:] - phi[:,:,:,:-2]) / (2*dS)
    dphi_dt = (phi[:,:,2:,:] - phi[:,:,:-2,:]) / (2*dt)
    dphi_dS_inner = dphi_dS[:,:,1:-1,:]
    dphi_dt_inner = dphi_dt[:,:,:,1:-1]
    return (dphi_dS_inner**2 + dphi_dt_inner**2).squeeze()


# ==================================================================
# Full 4-stage training pipeline
# ==================================================================
def train_pipeline(cfg, label="exp"):
    t0 = time.time()
    device = cfg.device
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    # FDM
    S_fdm, t_fdm, V_fdm = generate_fdm(cfg)
    boundary_fdm = extract_boundary(S_fdm, t_fdm, V_fdm, cfg.K)

    # PINN grid
    S_pinn = np.linspace(0, cfg.S_max, cfg.grid_S)
    t_pinn = np.linspace(0, cfg.T, cfg.grid_t)
    tt_np, ss_np = np.meshgrid(t_pinn, S_pinn, indexing='ij')

    fdm_interp = RegularGridInterpolator(
        (t_fdm, S_fdm), V_fdm, method='linear',
        bounds_error=False, fill_value=None)
    pts = np.stack([tt_np.ravel(), ss_np.ravel()], axis=-1)
    V_ref = fdm_interp(pts).reshape(cfg.grid_t, cfg.grid_S)

    phi_tgt_np = build_phi_target(cfg, S_fdm, t_fdm, V_fdm, S_pinn, t_pinn)

    # Tensors
    S_vals = torch.linspace(0, 1, cfg.grid_S, device=device)
    t_vals = torch.linspace(0, 1, cfg.grid_t, device=device)
    tt, ss = torch.meshgrid(t_vals, S_vals, indexing='ij')
    S_grid = ss.unsqueeze(0).unsqueeze(0)
    t_grid = tt.unsqueeze(0).unsqueeze(0)
    S_flat = ss.reshape(-1, 1); t_flat = tt.reshape(-1, 1)

    V_ref_t = torch.tensor(V_ref/cfg.K, dtype=torch.float32, device=device)
    V_ref_full = torch.tensor(V_ref, dtype=torch.float32, device=device)
    V_target_flat = V_ref_t.reshape(-1, 1)
    phi_tgt_t = torch.tensor(phi_tgt_np, dtype=torch.float32, device=device)
    phi_tgt_4d = phi_tgt_t.unsqueeze(0).unsqueeze(0)
    phi_anchor_flat = phi_tgt_t.reshape(-1, 1)
    payoff_grid = torch.clamp(cfg.K - ss*cfg.S_max, min=0.0) / cfg.K
    payoff_flat = payoff_grid.reshape(-1, 1)
    exercise_ratio = (phi_tgt_t < 0.5).float().mean().item()
    dS_norm = 1.0/(cfg.grid_S-1); dt_norm = 1.0/(cfg.grid_t-1)

    # Networks
    v_net = ValueMLP(cfg.mlp_hidden, cfg.mlp_layers, cfg.omega_0).to(device)
    phi_net = PhiUNet(cfg.cnn_in_channels, cfg.cnn_base_ch, cfg.cnn_levels).to(device)

    # === Stage 1 ===
    print(f"  [{label}] Stage 1 ({cfg.stage1_epochs} ep)...")
    opt1 = torch.optim.AdamW(v_net.parameters(), lr=cfg.stage1_lr, weight_decay=1e-5)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg.stage1_epochs, eta_min=1e-5)
    for ep in range(1, cfg.stage1_epochs+1):
        opt1.zero_grad()
        mse = F.mse_loss(v_net(S_flat, t_flat), V_target_flat)
        bc = compute_bc_loss(v_net, cfg)
        (mse + cfg.lambda_bc*bc).backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
        opt1.step(); sch1.step()
    s1_rel = compute_rel_err(v_net, S_flat, t_flat, V_ref_full, cfg)
    print(f"    Done. RelErr={s1_rel:.6f}")

    # === Stage 1.5 ===
    print(f"  [{label}] Stage 1.5 ({cfg.stage15_epochs} ep)...")
    opt15 = torch.optim.Adam(phi_net.parameters(), lr=cfg.stage15_lr)
    sch15 = torch.optim.lr_scheduler.CosineAnnealingLR(opt15, T_max=cfg.stage15_epochs, eta_min=1e-5)
    with torch.no_grad():
        V_det = v_net(S_flat, t_flat).reshape(1,1,cfg.grid_t,cfg.grid_S)
    cnn0 = torch.cat([S_grid, t_grid, V_det.detach()], dim=1)
    for ep in range(1, cfg.stage15_epochs+1):
        opt15.zero_grad()
        bce = F.binary_cross_entropy(phi_net(cnn0), phi_tgt_4d)
        bce.backward(); opt15.step(); sch15.step()
    print(f"    Done. BCE={bce.item():.6f}")

    # === Stage 2 ===
    print(f"  [{label}] Stage 2 ({cfg.stage2_epochs} ep)...")
    opt_v = torch.optim.AdamW(v_net.parameters(), lr=cfg.stage2_lr_mlp, weight_decay=1e-5)
    opt_p = torch.optim.AdamW(phi_net.parameters(), lr=cfg.stage2_lr_cnn, weight_decay=1e-5)
    sch_v = torch.optim.lr_scheduler.CosineAnnealingLR(opt_v, T_max=cfg.stage2_epochs, eta_min=1e-6)
    sch_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=cfg.stage2_epochs, eta_min=1e-5)

    for epoch in range(1, cfg.stage2_epochs+1):
        v_net.train(); phi_net.train()
        S_n = S_flat.detach().requires_grad_(True)
        t_n = t_flat.detach().requires_grad_(True)
        V_norm_pred, bs_res = compute_bs_residual(v_net, S_n, t_n, cfg)

        V_map = V_norm_pred.detach().reshape(1,1,cfg.grid_t,cfg.grid_S)
        cnn_in = torch.cat([S_grid, t_grid, V_map], dim=1)
        phi_2d = phi_net(cnn_in).squeeze()
        phi_fl = phi_2d.reshape(-1, 1)

        L_pde = (phi_fl * bs_res**2).mean()
        L_ex = ((1-phi_fl)**2*(V_norm_pred-payoff_flat)**2).mean()/cfg.eps
        phi_inner = phi_2d[1:-1,1:-1]
        W_phi = phi_inner**2*(1-phi_inner)**2
        grad_sq = phi_gradient_fd_inner(phi_2d, dS_norm, dt_norm)
        L_int = (cfg.eps*grad_sq + W_phi/cfg.eps).mean()
        L_bc = compute_bc_loss(v_net, cfg)
        L_anc = focal_bce(phi_fl, phi_anchor_flat, gamma=2.0)
        L_bal = (phi_2d.mean() - (1.0-exercise_ratio))**2

        if epoch < cfg.anchor_decay_start:
            anc_w = cfg.lambda_anchor
        elif epoch >= cfg.anchor_decay_end:
            anc_w = cfg.lambda_anchor * cfg.anchor_min_ratio
        else:
            prog = (epoch-cfg.anchor_decay_start)/(cfg.anchor_decay_end-cfg.anchor_decay_start)
            anc_w = cfg.lambda_anchor*(1.0-prog*(1.0-cfg.anchor_min_ratio))

        L_tot = (cfg.lambda_pde*L_pde + cfg.lambda_ex*L_ex + cfg.lambda_int*L_int
                 + cfg.lambda_bc*L_bc + anc_w*L_anc + cfg.lambda_balance*L_bal)

        opt_v.zero_grad(); opt_p.zero_grad(); L_tot.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(phi_net.parameters(), 1.0)
        opt_v.step(); opt_p.step(); sch_v.step(); sch_p.step()

        if epoch % 5000 == 0:
            rel = compute_rel_err(v_net, S_flat, t_flat, V_ref_full, cfg)
            print(f"    Ep {epoch}: RelErr={rel:.4f}")

    s2_rel = compute_rel_err(v_net, S_flat, t_flat, V_ref_full, cfg)
    print(f"    Done. RelErr={s2_rel:.6f}")

    # === Stage 3 ===
    print(f"  [{label}] Stage 3 ({cfg.stage3_steps} steps)...")
    phi_net.eval()
    for p in phi_net.parameters(): p.requires_grad_(False)
    with torch.no_grad():
        V_map3 = v_net(S_flat, t_flat).reshape(1,1,cfg.grid_t,cfg.grid_S)
        cnn3 = torch.cat([S_grid, t_grid, V_map3], dim=1)
        phi_fixed = phi_net(cnn3).squeeze()
        phi_fixed_flat = phi_fixed.reshape(-1,1)

    lbfgs = torch.optim.LBFGS(v_net.parameters(), lr=0.5, max_iter=5,
                               history_size=20, line_search_fn='strong_wolfe')
    for step in range(cfg.stage3_steps):
        def closure():
            lbfgs.zero_grad()
            S_n2 = S_flat.detach().requires_grad_(True)
            t_n2 = t_flat.detach().requires_grad_(True)
            V_n2, res2 = compute_bs_residual(v_net, S_n2, t_n2, cfg)
            Lp = (phi_fixed_flat*res2**2).mean()
            Le = ((1-phi_fixed_flat)**2*(V_n2-payoff_flat)**2).mean()/cfg.eps
            Lb = compute_bc_loss(v_net, cfg)
            loss = cfg.lambda_pde*Lp + cfg.lambda_ex*Le + cfg.lambda_bc*Lb
            loss.backward(); return loss
        lbfgs.step(closure)

    s3_rel = compute_rel_err(v_net, S_flat, t_flat, V_ref_full, cfg)
    elapsed = time.time() - t0
    print(f"    Done. RelErr={s3_rel:.6f}, Time={elapsed:.1f}s")

    with torch.no_grad():
        V_final = (v_net(S_flat, t_flat).reshape(cfg.grid_t,cfg.grid_S)*cfg.K).cpu().numpy()
        phi_np = phi_fixed.cpu().numpy()

    # Free boundary
    fb_pred = []
    for i in range(cfg.grid_t):
        row = phi_np[i,:]
        idx = np.where(row[:-1] < 0.5)[0]
        if len(idx) > 0:
            j = idx[-1]
            if j < cfg.grid_S-1 and row[j+1] != row[j]:
                s_fb = S_pinn[j] + (0.5-row[j])/(row[j+1]-row[j])*(S_pinn[j+1]-S_pinn[j])
            else: s_fb = S_pinn[j]
            fb_pred.append(s_fb)
        else: fb_pred.append(0.0)
    fb_pred = np.array(fb_pred)
    fb_fdm = np.interp(t_pinn, t_fdm, boundary_fdm)
    mask = fb_fdm > 0
    fb_mean = np.mean(np.abs(fb_pred[mask]-fb_fdm[mask])) if mask.sum()>0 else 0.0
    fb_rel = fb_mean/np.mean(fb_fdm[mask]) if mask.sum()>0 else 0.0

    return {
        'final_relerr': s3_rel, 'stage1_relerr': s1_rel,
        'stage2_relerr': s2_rel,
        'phi_mean': float(phi_np.mean()), 'phi_std': float(phi_np.std()),
        'phi_min': float(phi_np.min()), 'phi_max': float(phi_np.max()),
        'exercise_frac': float((phi_np<0.1).mean()),
        'continuation_frac': float((phi_np>0.9).mean()),
        'fb_mean_err': float(fb_mean), 'fb_rel': float(fb_rel),
        'time': elapsed, 'epsilon': cfg.eps,
        'phi': phi_np, 'V': V_final,
    }


# ==================================================================
# PART 1: Stage 2 φ snapshots
# ==================================================================
def run_snapshots():
    print("\n" + "="*70)
    print("PART 1: φ snapshots (baseline ε=0.05)")
    print("="*70)

    cfg = Config(); device = cfg.device
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    S_fdm, t_fdm, V_fdm = generate_fdm(cfg)
    S_pinn = np.linspace(0, cfg.S_max, cfg.grid_S)
    t_pinn = np.linspace(0, cfg.T, cfg.grid_t)
    tt_np, ss_np = np.meshgrid(t_pinn, S_pinn, indexing='ij')

    fdm_interp = RegularGridInterpolator(
        (t_fdm, S_fdm), V_fdm, method='linear',
        bounds_error=False, fill_value=None)
    pts = np.stack([tt_np.ravel(), ss_np.ravel()], axis=-1)
    V_ref = fdm_interp(pts).reshape(cfg.grid_t, cfg.grid_S)
    phi_tgt_np = build_phi_target(cfg, S_fdm, t_fdm, V_fdm, S_pinn, t_pinn)

    S_vals = torch.linspace(0,1,cfg.grid_S,device=device)
    t_vals = torch.linspace(0,1,cfg.grid_t,device=device)
    tt, ss = torch.meshgrid(t_vals, S_vals, indexing='ij')
    S_grid = ss.unsqueeze(0).unsqueeze(0)
    t_grid = tt.unsqueeze(0).unsqueeze(0)
    S_flat = ss.reshape(-1,1); t_flat = tt.reshape(-1,1)

    V_ref_t = torch.tensor(V_ref/cfg.K, dtype=torch.float32, device=device)
    V_ref_full = torch.tensor(V_ref, dtype=torch.float32, device=device)
    V_target_flat = V_ref_t.reshape(-1,1)
    phi_tgt_t = torch.tensor(phi_tgt_np, dtype=torch.float32, device=device)
    phi_tgt_4d = phi_tgt_t.unsqueeze(0).unsqueeze(0)
    phi_anchor_flat = phi_tgt_t.reshape(-1,1)
    payoff_grid = torch.clamp(cfg.K - ss*cfg.S_max, min=0.0)/cfg.K
    payoff_flat = payoff_grid.reshape(-1,1)
    exercise_ratio = (phi_tgt_t<0.5).float().mean().item()
    dS_norm = 1.0/(cfg.grid_S-1); dt_norm = 1.0/(cfg.grid_t-1)

    v_net = ValueMLP(cfg.mlp_hidden, cfg.mlp_layers, cfg.omega_0).to(device)
    phi_net = PhiUNet(cfg.cnn_in_channels, cfg.cnn_base_ch, cfg.cnn_levels).to(device)

    # Stage 1
    print("  Stage 1...")
    opt1 = torch.optim.AdamW(v_net.parameters(), lr=cfg.stage1_lr, weight_decay=1e-5)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg.stage1_epochs, eta_min=1e-5)
    for ep in range(1, cfg.stage1_epochs+1):
        opt1.zero_grad()
        mse = F.mse_loss(v_net(S_flat, t_flat), V_target_flat)
        bc = compute_bc_loss(v_net, cfg)
        (mse + cfg.lambda_bc*bc).backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
        opt1.step(); sch1.step()
    print("    Done.")

    # Stage 1.5
    print("  Stage 1.5...")
    opt15 = torch.optim.Adam(phi_net.parameters(), lr=cfg.stage15_lr)
    sch15 = torch.optim.lr_scheduler.CosineAnnealingLR(opt15, T_max=cfg.stage15_epochs, eta_min=1e-5)
    with torch.no_grad():
        V_det = v_net(S_flat, t_flat).reshape(1,1,cfg.grid_t,cfg.grid_S)
    cnn0 = torch.cat([S_grid, t_grid, V_det.detach()], dim=1)
    for ep in range(1, cfg.stage15_epochs+1):
        opt15.zero_grad()
        bce = F.binary_cross_entropy(phi_net(cnn0), phi_tgt_4d)
        bce.backward(); opt15.step(); sch15.step()
    print("    Done.")

    # Stage 2 with snapshots
    print("  Stage 2 with snapshots...")
    SNAP_EP = [1, 100, 500, 1000, 3000, 5000, 10000, 20000]
    snap_phis = {}; snap_meta = {}

    opt_v = torch.optim.AdamW(v_net.parameters(), lr=cfg.stage2_lr_mlp, weight_decay=1e-5)
    opt_p = torch.optim.AdamW(phi_net.parameters(), lr=cfg.stage2_lr_cnn, weight_decay=1e-5)
    sch_v = torch.optim.lr_scheduler.CosineAnnealingLR(opt_v, T_max=cfg.stage2_epochs, eta_min=1e-6)
    sch_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=cfg.stage2_epochs, eta_min=1e-5)

    for epoch in range(1, cfg.stage2_epochs+1):
        v_net.train(); phi_net.train()
        S_n = S_flat.detach().requires_grad_(True)
        t_n = t_flat.detach().requires_grad_(True)
        V_norm_pred, bs_res = compute_bs_residual(v_net, S_n, t_n, cfg)

        V_map = V_norm_pred.detach().reshape(1,1,cfg.grid_t,cfg.grid_S)
        cnn_in = torch.cat([S_grid, t_grid, V_map], dim=1)
        phi_2d = phi_net(cnn_in).squeeze()
        phi_fl = phi_2d.reshape(-1,1)

        L_pde = (phi_fl*bs_res**2).mean()
        L_ex = ((1-phi_fl)**2*(V_norm_pred-payoff_flat)**2).mean()/cfg.eps
        phi_inner = phi_2d[1:-1,1:-1]
        W_phi = phi_inner**2*(1-phi_inner)**2
        grad_sq = phi_gradient_fd_inner(phi_2d, dS_norm, dt_norm)
        L_int = (cfg.eps*grad_sq + W_phi/cfg.eps).mean()
        L_bc = compute_bc_loss(v_net, cfg)
        L_anc = focal_bce(phi_fl, phi_anchor_flat, gamma=2.0)
        L_bal = (phi_2d.mean()-(1.0-exercise_ratio))**2

        if epoch < cfg.anchor_decay_start: anc_w = cfg.lambda_anchor
        elif epoch >= cfg.anchor_decay_end: anc_w = cfg.lambda_anchor*cfg.anchor_min_ratio
        else:
            prog = (epoch-cfg.anchor_decay_start)/(cfg.anchor_decay_end-cfg.anchor_decay_start)
            anc_w = cfg.lambda_anchor*(1.0-prog*(1.0-cfg.anchor_min_ratio))

        L_tot = (cfg.lambda_pde*L_pde + cfg.lambda_ex*L_ex + cfg.lambda_int*L_int
                 + cfg.lambda_bc*L_bc + anc_w*L_anc + cfg.lambda_balance*L_bal)

        opt_v.zero_grad(); opt_p.zero_grad(); L_tot.backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(phi_net.parameters(), 1.0)
        opt_v.step(); opt_p.step(); sch_v.step(); sch_p.step()

        if epoch in SNAP_EP:
            with torch.no_grad():
                phi_s = phi_net(cnn_in).squeeze().cpu().numpy().copy()
                rel = compute_rel_err(v_net, S_flat, t_flat, V_ref_full, cfg)
            snap_phis[epoch] = phi_s
            snap_meta[epoch] = {'rel_err': rel,
                                'phi_mean': float(phi_s.mean()),
                                'phi_std': float(phi_s.std())}
            print(f"    [Snap] Ep {epoch:>5d} | RelErr={rel:.4f} | "
                  f"φ mean={phi_s.mean():.3f} std={phi_s.std():.3f}")

    save_dict = {'S': S_pinn, 't': t_pinn,
                 'epochs': np.array(SNAP_EP), 'phi_target': phi_tgt_np}
    for ep in SNAP_EP:
        save_dict[f'phi_{ep}'] = snap_phis[ep]
    np.savez(os.path.join(OUT_DIR, 'phi_snapshots.npz'), **save_dict)
    with open(os.path.join(OUT_DIR, 'phi_snapshots_meta.json'), 'w') as f:
        json.dump({str(k): v for k, v in snap_meta.items()}, f, indent=2)
    print("  Snapshots saved.\n")


# ==================================================================
# PART 2: ε sensitivity
# ==================================================================
def run_epsilon_sensitivity():
    print("\n" + "="*70)
    print("PART 2: ε sensitivity study")
    print("="*70)

    EPSILONS = [0.01, 0.02, 0.05, 0.10, 0.20]
    results = {}; arrays = {}

    for eps_val in EPSILONS:
        print(f"\n{'─'*60}\n  ε = {eps_val}\n{'─'*60}")

        if eps_val == 0.05:
            print("  Loading baseline from scheme1_results.npz...")
            data = np.load('scheme1_results.npz')
            phi_np = data['phi']; V_np = data['V_pred']
            S_pinn = data['S']; t_pinn = data['t']

            fb_pred = []
            for i in range(len(t_pinn)):
                row = phi_np[i,:]
                idx = np.where(row[:-1]<0.5)[0]
                if len(idx)>0:
                    j=idx[-1]
                    if j<len(S_pinn)-1 and row[j+1]!=row[j]:
                        s_fb=S_pinn[j]+(0.5-row[j])/(row[j+1]-row[j])*(S_pinn[j+1]-S_pinn[j])
                    else: s_fb=S_pinn[j]
                    fb_pred.append(s_fb)
                else: fb_pred.append(0.0)
            fb_pred = np.array(fb_pred)
            fb_fdm = np.interp(t_pinn, data['t_fdm'], data['fdm_boundary'])
            mask = fb_fdm>0
            fb_mean = np.mean(np.abs(fb_pred[mask]-fb_fdm[mask]))
            fb_rel = fb_mean/np.mean(fb_fdm[mask])

            results[eps_val] = {
                'final_relerr': 0.000376,
                'phi_mean': float(phi_np.mean()), 'phi_std': float(phi_np.std()),
                'exercise_frac': float((phi_np<0.1).mean()),
                'continuation_frac': float((phi_np>0.9).mean()),
                'fb_mean_err': float(fb_mean), 'fb_rel': float(fb_rel),
                'time': 4839.0, 'epsilon': 0.05,
            }
            arrays[eps_val] = {'phi': phi_np, 'V': V_np}
            print(f"  Loaded. RelErr=0.000376")
        else:
            cfg_new = Config(); cfg_new.eps = eps_val
            res = train_pipeline(cfg_new, label=f"eps={eps_val}")
            results[eps_val] = {k: res[k] for k in res if k not in ('phi','V')}
            arrays[eps_val] = {'phi': res['phi'], 'V': res['V']}

    # Save
    with open(os.path.join(OUT_DIR, 'epsilon_sensitivity.json'), 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    cfg_ref = Config()
    save_dict = {'S': np.linspace(0,cfg_ref.S_max,cfg_ref.grid_S),
                 't': np.linspace(0,cfg_ref.T,cfg_ref.grid_t),
                 'epsilons': np.array(EPSILONS)}
    for e in EPSILONS:
        tag = str(e).replace('.','p')
        save_dict[f'phi_{tag}'] = arrays[e]['phi']
        save_dict[f'V_{tag}'] = arrays[e]['V']
    np.savez(os.path.join(OUT_DIR, 'epsilon_sensitivity.npz'), **save_dict)

    # Print
    print("\n" + "="*90 + "\nε SENSITIVITY SUMMARY\n" + "="*90)
    print(f"{'ε':>8s}  {'RelErr':>10s}  {'φ mean':>8s}  {'φ std':>8s}  "
          f"{'Ex%':>6s}  {'Cont%':>6s}  {'FB rel':>8s}  {'Time':>8s}")
    print("-"*90)
    for e in EPSILONS:
        r = results[e]
        print(f"{e:>8.2f}  {r['final_relerr']:>10.6f}  "
              f"{r['phi_mean']:>8.3f}  {r['phi_std']:>8.3f}  "
              f"{r.get('exercise_frac',0)*100:>5.1f}%  "
              f"{r.get('continuation_frac',0)*100:>5.1f}%  "
              f"{r['fb_rel']:>8.4f}  {r['time']:>8.1f}")

    print("\n% === LaTeX table ===")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Sensitivity to phase-field width $\varepsilon$.}")
    print(r"\label{tab:epsilon}")
    print(r"\begin{tabular}{@{}cccccr@{}}")
    print(r"\toprule")
    print(r"$\varepsilon$ & RelErr & $\bar{\varphi}$ & "
          r"$\sigma_\varphi$ & FB rel. & Time (s) \\")
    print(r"\midrule")
    for e in EPSILONS:
        r = results[e]
        star = r" $\star$" if e==0.05 else ""
        print(f"${e}${star} & ${r['final_relerr']*100:.2f}\\%$ & "
              f"${r['phi_mean']:.3f}$ & ${r['phi_std']:.3f}$ & "
              f"${r['fb_rel']*100:.2f}\\%$ & ${r['time']:.0f}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("\nDone.\n")


# ==================================================================
if __name__ == '__main__':
    t_start = time.time()
    print("="*70 + "\nAppendix C: Data Generation\n" + "="*70)
    run_snapshots()
    run_epsilon_sensitivity()
    elapsed = time.time()-t_start
    print(f"\n{'='*70}\nALL DONE. Total: {elapsed/3600:.1f} hours")
    print(f"Data in: {OUT_DIR}/")
    for fn in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, fn))
        print(f"  {fn} ({sz/1024:.1f} KB)")
    print("="*70 + "\nNext: python appendix_plot_figures.py")