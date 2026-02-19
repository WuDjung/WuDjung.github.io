#!/usr/bin/env python3
"""
appendix_plot_figures.py
=========================
Generate all Appendix C figures from pre-computed data.
Reads from appendix_data/, writes to figures_appendix/.

Usage:
    cd phi_cnn_mlp
    python appendix_plot_figures.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

DATA_DIR = 'appendix_data'
FIG_DIR = 'figures_appendix'
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'))
    fig.savefig(os.path.join(FIG_DIR, f'{name}.pdf'))
    plt.close(fig)
    print(f"  Saved {name}.{{png,pdf}}")


# ==================================================================
# Fig C.1: φ evolution
# ==================================================================
def fig_c1():
    print("Fig C.1: φ evolution...")
    data = np.load(os.path.join(DATA_DIR, 'phi_snapshots.npz'))
    with open(os.path.join(DATA_DIR, 'phi_snapshots_meta.json')) as f:
        meta = json.load(f)

    S = data['S']; t = data['t']
    SS, TT = np.meshgrid(S, t)
    epochs_show = [1, 500, 1000, 5000, 10000, 20000]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for i, ep in enumerate(epochs_show):
        ax = axes[i]
        phi = data[f'phi_{ep}']
        m = meta[str(ep)]
        im = ax.pcolormesh(SS, TT, phi, vmin=0, vmax=1,
                           cmap='RdYlBu_r', shading='auto')
        ax.set_title(
            f'Epoch {ep}\n'
            f'$\\bar{{\\varphi}}$={m["phi_mean"]:.3f}, '
            f'$\\sigma_\\varphi$={m["phi_std"]:.3f}, '
            f'RelErr={m["rel_err"]:.4f}', fontsize=10)
        ax.set_xlabel('$S$'); ax.set_ylabel('$t$')
        ax.set_xlim(0, 150)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        'Evolution of $\\varphi$ during Stage 2 joint training',
        fontsize=14, y=1.01)
    plt.tight_layout()
    save_fig(fig, 'figC1_phi_evolution')


# ==================================================================
# Fig C.2: φ cross-sections
# ==================================================================
def fig_c2():
    print("Fig C.2: φ cross-sections...")
    data = np.load(os.path.join(DATA_DIR, 'phi_snapshots.npz'))
    S = data['S']; t = data['t']
    t_idx = np.argmin(np.abs(t - 0.5))

    epochs_show = [1, 500, 1000, 5000, 10000, 20000]
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(epochs_show) - 1))
              for i in range(len(epochs_show))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, ep in enumerate(epochs_show):
        phi = data[f'phi_{ep}']
        ax1.plot(S, phi[t_idx, :], color=colors[i],
                 label=f'Epoch {ep}', linewidth=1.5)

    ax1.axhline(0.5, color='gray', ls='--', lw=0.8, label='$\\varphi=0.5$')
    ax1.set_xlabel('$S$'); ax1.set_ylabel('$\\varphi$')
    ax1.set_title(f'$\\varphi(S)$ profiles at $t={t[t_idx]:.2f}$')
    ax1.set_xlim(50, 130); ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    for i, ep in enumerate(epochs_show):
        phi = data[f'phi_{ep}'].ravel()
        ax2.hist(phi, bins=50, alpha=0.3, color=colors[i],
                 label=f'Epoch {ep}', density=True)
    ax2.set_xlabel('$\\varphi$'); ax2.set_ylabel('Density')
    ax2.set_title('Distribution of $\\varphi$ during training')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'figC2_phi_cross_sections')


# ==================================================================
# Fig C.3: Robustness bar chart
# ==================================================================
def fig_c3():
    print("Fig C.3: Robustness comparison...")
    with open('robustness_results.json') as f:
        rob = json.load(f)

    cases = ['base', 'low_vol', 'high_vol', 'low_rate', 'short_mat']
    labels = ['Base\n$\\sigma$=0.2\nr=0.05, T=1',
              'Low Vol\n$\\sigma$=0.1',
              'High Vol\n$\\sigma$=0.4',
              'Low Rate\nr=0.02',
              'Short Mat\nT=0.25']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    relerrs = [rob[c]['final_relerr'] * 100 for c in cases]
    phi_means = [rob[c]['phi_mean'] for c in cases]
    phi_stds = [rob[c]['phi_std'] for c in cases]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars = ax1.bar(range(len(cases)), relerrs, color=colors,
                   edgecolor='black', linewidth=0.5, width=0.6)
    ax1.set_xticks(range(len(cases)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('Relative $L^2$ Error (%)')
    ax1.set_title('Pricing Accuracy Across Parameter Regimes')
    ax1.set_ylim(0, max(relerrs) * 1.8)
    ax1.axhline(0.05, color='red', ls='--', lw=0.8, label='0.05% threshold')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, relerrs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.3f}%', ha='center', va='bottom', fontsize=9)

    ax2.bar(range(len(cases)), phi_means, yerr=phi_stds, color=colors,
            edgecolor='black', linewidth=0.5, width=0.6, capsize=5, alpha=0.8)
    ax2.set_xticks(range(len(cases)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('$\\bar{\\varphi}$ (mean $\\pm$ std)')
    ax2.set_title('Phase-Field Statistics')
    ax2.set_ylim(0, 1.1); ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_fig(fig, 'figC3_robustness')


# ==================================================================
# Fig C.4: Greeks comparison
# ==================================================================
def fig_c4():
    print("Fig C.4: Greeks comparison...")
    from appendix_run_experiments import Config, ValueMLP, thomas_solve

    cfg = Config()

    v_net = ValueMLP(cfg.mlp_hidden, cfg.mlp_layers, cfg.omega_0).to(cfg.device)
    ckpt = torch.load('scheme1_best.pth', map_location='cpu')
    v_net.load_state_dict(ckpt['v_net'])
    v_net.eval()

    # FDM reference
    S_max, T, K, r, sigma = cfg.S_max, cfg.T, cfg.K, cfg.r, cfg.sigma
    Ns, Nt = cfg.fdm_S, cfg.fdm_t
    sigma2 = sigma**2; dt = T/Nt
    S_fdm_arr = np.linspace(0, S_max, Ns+1)
    payoff = np.maximum(K - S_fdm_arr, 0.0)
    V = payoff.copy()
    V_all = np.zeros((Nt+1, Ns+1)); V_all[Nt,:] = V.copy()
    for n in range(Nt-1, -1, -1):
        a=np.zeros(Ns+1); b=np.zeros(Ns+1)
        c_arr=np.zeros(Ns+1); d=np.zeros(Ns+1)
        for j in range(1,Ns):
            a[j]=0.5*dt*(r*j-sigma2*j**2)
            b[j]=1+dt*(sigma2*j**2+r)
            c_arr[j]=-0.5*dt*(r*j+sigma2*j**2)
            d[j]=V[j]
        b[0]=1.0; d[0]=K; b[Ns]=1.0; d[Ns]=0.0
        V_new=thomas_solve(a,b,c_arr,d,Ns+1)
        V_new=np.maximum(V_new,payoff)
        V=V_new.copy(); V_all[n,:]=V.copy()
    t_fdm_arr = np.linspace(0, T, Nt+1)

    t_val = 0.5

    # NN Greeks
    S_dense = np.linspace(1, 180, 500)
    St = torch.tensor(S_dense/cfg.S_max, dtype=torch.float32,
                      device=cfg.device).reshape(-1,1).requires_grad_(True)
    tt_g = torch.full_like(St, t_val/cfg.T).requires_grad_(True)
    Vout = v_net(St, tt_g)

    dVdS = torch.autograd.grad(Vout, St,
        grad_outputs=torch.ones_like(Vout), create_graph=True)[0]
    Delta_nn = (dVdS * cfg.K/cfg.S_max).detach().cpu().numpy().ravel()

    d2VdS2 = torch.autograd.grad(dVdS, St,
        grad_outputs=torch.ones_like(dVdS), create_graph=True)[0]
    Gamma_nn = (d2VdS2 * cfg.K/cfg.S_max**2).detach().cpu().numpy().ravel()

    dVdt = torch.autograd.grad(Vout, tt_g,
        grad_outputs=torch.ones_like(Vout), create_graph=False)[0]
    Theta_nn = (dVdt * cfg.K/cfg.T).detach().cpu().numpy().ravel()

    # FDM Greeks
    t_idx = np.argmin(np.abs(t_fdm_arr - t_val))
    V_slice = V_all[t_idx, :]
    dS = S_fdm_arr[1]-S_fdm_arr[0]
    Delta_fdm = np.gradient(V_slice, dS)
    Gamma_fdm = np.gradient(Delta_fdm, dS)
    dt_g2 = t_fdm_arr[1]-t_fdm_arr[0]
    ti = t_idx
    Theta_fdm = (V_all[min(ti+1,Nt),:]-V_all[max(ti-1,0),:])/(2*dt_g2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, nn, fdm, name in zip(
        axes,
        [Delta_nn, Gamma_nn, Theta_nn],
        [Delta_fdm, Gamma_fdm, Theta_fdm],
        ['$\\Delta = \\partial V/\\partial S$',
         '$\\Gamma = \\partial^2 V/\\partial S^2$',
         '$\\Theta = \\partial V/\\partial t$']
    ):
        ax.plot(S_dense, nn, 'r-', lw=1.5, label='NN (autograd)')
        ax.plot(S_fdm_arr, fdm, 'b--', lw=1.0, alpha=0.7,
                label='FDM (finite-diff)')
        ax.set_xlabel('$S$'); ax.set_ylabel(name)
        ax.set_title(f'{name} at $t={t_val}$')
        ax.set_xlim(50, 150); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'figC4_greeks')


# ==================================================================
# Fig C.5: ε sensitivity
# ==================================================================
def fig_c5():
    print("Fig C.5: ε sensitivity...")
    with open(os.path.join(DATA_DIR, 'epsilon_sensitivity.json')) as f:
        eps_data = json.load(f)
    data = np.load(os.path.join(DATA_DIR, 'epsilon_sensitivity.npz'))

    S = data['S']; t = data['t']
    epsilons = sorted([float(k) for k in eps_data.keys()])

    # Panel A: metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    relerrs = [eps_data[str(e)]['final_relerr']*100 for e in epsilons]
    axes[0].plot(epsilons, relerrs, 'bo-', lw=2, markersize=8)
    axes[0].axvline(0.05, color='red', ls='--', lw=0.8,
                    label='$\\varepsilon=0.05$ (paper)')
    axes[0].set_xlabel('$\\varepsilon$')
    axes[0].set_ylabel('Relative $L^2$ Error (%)')
    axes[0].set_title('Pricing Error vs $\\varepsilon$')
    axes[0].set_xscale('log'); axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    for e, r in zip(epsilons, relerrs):
        axes[0].annotate(f'{r:.3f}%', (e, r), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=9)

    phi_stds = [eps_data[str(e)]['phi_std'] for e in epsilons]
    axes[1].plot(epsilons, phi_stds, 'gs-', lw=2, markersize=8)
    axes[1].axvline(0.05, color='red', ls='--', lw=0.8)
    axes[1].set_xlabel('$\\varepsilon$')
    axes[1].set_ylabel('$\\sigma_\\varphi$')
    axes[1].set_title('Phase-Field Sharpness vs $\\varepsilon$')
    axes[1].set_xscale('log'); axes[1].grid(True, alpha=0.3)

    fb_rels = [eps_data[str(e)]['fb_rel']*100 for e in epsilons]
    axes[2].plot(epsilons, fb_rels, 'r^-', lw=2, markersize=8)
    axes[2].axvline(0.05, color='red', ls='--', lw=0.8)
    axes[2].set_xlabel('$\\varepsilon$')
    axes[2].set_ylabel('FB Relative Error (%)')
    axes[2].set_title('Free Boundary Error vs $\\varepsilon$')
    axes[2].set_xscale('log'); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'figC5_epsilon_metrics')

    # Panel B: φ heatmaps
    SS, TT = np.meshgrid(S, t)
    fig2, axes2 = plt.subplots(1, len(epsilons),
                                figsize=(4*len(epsilons), 4))

    for i, e in enumerate(epsilons):
        ax = axes2[i]
        tag = str(e).replace('.', 'p')
        phi = data[f'phi_{tag}']
        im = ax.pcolormesh(SS, TT, phi, vmin=0, vmax=1,
                           cmap='RdYlBu_r', shading='auto')
        star = ' $\\star$' if e == 0.05 else ''
        ax.set_title(
            f'$\\varepsilon={e}${star}\n'
            f'RelErr={eps_data[str(e)]["final_relerr"]*100:.3f}%',
            fontsize=10)
        ax.set_xlabel('$S$')
        if i == 0: ax.set_ylabel('$t$')
        ax.set_xlim(0, 150)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig2.suptitle(
        'Phase-field $\\varphi$ for different $\\varepsilon$ values',
        fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig2, 'figC5_epsilon_heatmaps')


# ==================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Appendix C: Figure Generation")
    print("=" * 70)

    required = ['phi_snapshots.npz', 'phi_snapshots_meta.json',
                'epsilon_sensitivity.json', 'epsilon_sensitivity.npz']
    missing = [f for f in required
               if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print(f"ERROR: Missing: {missing}")
        print("Run 'python appendix_run_experiments.py' first.")
        sys.exit(1)

    fig_c1()
    fig_c2()
    fig_c3()
    fig_c4()
    fig_c5()

    print("\n" + "=" * 70)
    print(f"All figures in {FIG_DIR}/")
    for fn in sorted(os.listdir(FIG_DIR)):
        sz = os.path.getsize(os.path.join(FIG_DIR, fn))
        print(f"  {fn} ({sz/1024:.1f} KB)")
    print("=" * 70)