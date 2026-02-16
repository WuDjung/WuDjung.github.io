"""
scheme1_visualize.py
====================
Phase-Field + PINN 美式看跌期权 —— 可视化脚本
读取 scheme1_results.npz 和 scheme1_history.json，生成论文级图表。
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
import os

# ============================================================
# 0. 全局绘图设置
# ============================================================
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

OUTPUT_DIR = "figures1"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """加载结果文件"""
    data = np.load("scheme1_results.npz", allow_pickle=True)
    with open("scheme1_history.json", "r") as f:
        history = json.load(f)
    return data, history


def add_colorbar(fig, ax, im, label="", size="4%", pad=0.08):
    """为 ax 添加紧凑的 colorbar"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=9)
    return cb


# ============================================================
# 图 1: V_pred vs V_fdm 热力图 + 误差图 + 自由边界叠加
# ============================================================
def plot_fig1_value_comparison(data):
    """三面板：V_FDM, V_PINN, |误差| + 自由边界叠加"""
    S = data["S"]             # (grid_S,)
    t = data["t"]             # (grid_t,)
    V_pred = data["V_pred"]   # (grid_t, grid_S)
    V_fdm = data["V_fdm"]    # (grid_t, grid_S)
    fb = data["fdm_boundary"] # (grid_t,) 或 (fdm_t,)

    # 若 fdm_boundary 长度与 t 不同，插值到 PINN 网格
    if len(fb) != len(t):
        t_fb = np.linspace(0, t[-1], len(fb))
        fb_interp = interp1d(t_fb, fb, kind='linear', fill_value='extrapolate')(t)
    else:
        fb_interp = fb

    abs_err = np.abs(V_pred - V_fdm)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    SS, TT = np.meshgrid(S, t)

    # --- V_FDM ---
    im0 = axes[0].pcolormesh(SS, TT, V_fdm, cmap='RdYlBu_r', shading='auto')
    axes[0].plot(fb_interp, t, 'k-', linewidth=2.0, label='Free boundary (FDM)')
    axes[0].set_title(r'$V_{\mathrm{FDM}}(S, t)$')
    axes[0].set_xlabel(r'$S$')
    axes[0].set_ylabel(r'$t$')
    axes[0].legend(loc='upper right', fontsize=9)
    add_colorbar(fig, axes[0], im0, label='Option value')

    # --- V_PINN ---
    im1 = axes[1].pcolormesh(SS, TT, V_pred, cmap='RdYlBu_r', shading='auto',
                              vmin=im0.get_clim()[0], vmax=im0.get_clim()[1])
    axes[1].plot(fb_interp, t, 'k-', linewidth=2.0, label='Free boundary (FDM)')
    axes[1].set_title(r'$V_{\mathrm{PINN}}(S, t)$')
    axes[1].set_xlabel(r'$S$')
    axes[1].set_ylabel(r'$t$')
    axes[1].legend(loc='upper right', fontsize=9)
    add_colorbar(fig, axes[1], im1, label='Option value')

    # --- |V_pred - V_fdm| ---
    vmax_err = np.percentile(abs_err, 99)
    im2 = axes[2].pcolormesh(SS, TT, abs_err, cmap='hot_r', shading='auto',
                              vmin=0, vmax=vmax_err)
    axes[2].plot(fb_interp, t, 'k--', linewidth=2.0, label='Free boundary (FDM)')
    axes[2].set_title(r'$|V_{\mathrm{PINN}} - V_{\mathrm{FDM}}|$')
    axes[2].set_xlabel(r'$S$')
    axes[2].set_ylabel(r'$t$')
    axes[2].legend(loc='upper right', fontsize=9)
    add_colorbar(fig, axes[2], im2, label='Absolute error')

    fig.suptitle('Figure 1: Value Function Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig1_value_comparison.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "fig1_value_comparison.png"))
    plt.close(fig)
    print("[Fig 1] Value comparison saved.")


# ============================================================
# 图 2: φ 热力图 + φ_target 对比 + 自由边界叠加
# ============================================================
def plot_fig2_phi_analysis(data):
    """三面板：φ_target, φ_pred, |φ差异| + 自由边界"""
    S = data["S"]
    t = data["t"]
    phi = data["phi"]              # (grid_t, grid_S)
    phi_target = data["phi_target"] # (grid_t, grid_S)
    fb = data["fdm_boundary"]

    if len(fb) != len(t):
        t_fb = np.linspace(0, t[-1], len(fb))
        fb_interp = interp1d(t_fb, fb, kind='linear', fill_value='extrapolate')(t)
    else:
        fb_interp = fb

    phi_diff = np.abs(phi - phi_target)
    SS, TT = np.meshgrid(S, t)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- φ_target ---
    im0 = axes[0].pcolormesh(SS, TT, phi_target, cmap='coolwarm', shading='auto',
                              vmin=0, vmax=1)
    axes[0].plot(fb_interp, t, 'k-', linewidth=2.0, label='Free boundary (FDM)')
    axes[0].set_title(r'$\phi_{\mathrm{target}}(S, t)$')
    axes[0].set_xlabel(r'$S$')
    axes[0].set_ylabel(r'$t$')
    axes[0].legend(loc='upper right', fontsize=9)
    add_colorbar(fig, axes[0], im0, label=r'$\phi$')

    # --- φ_pred ---
    im1 = axes[1].pcolormesh(SS, TT, phi, cmap='coolwarm', shading='auto',
                              vmin=0, vmax=1)
    axes[1].plot(fb_interp, t, 'k-', linewidth=2.0, label='Free boundary (FDM)')
    axes[1].set_title(r'$\phi_{\mathrm{pred}}(S, t)$')
    axes[1].set_xlabel(r'$S$')
    axes[1].set_ylabel(r'$t$')
    axes[1].legend(loc='upper right', fontsize=9)
    add_colorbar(fig, axes[1], im1, label=r'$\phi$')

    # --- |φ_pred - φ_target| ---
    im2 = axes[2].pcolormesh(SS, TT, phi_diff, cmap='magma', shading='auto',
                              vmin=0, vmax=np.percentile(phi_diff, 99))
    axes[2].plot(fb_interp, t, 'k--', linewidth=2.0, label='Free boundary (FDM)')
    axes[2].set_title(r'$|\phi_{\mathrm{pred}} - \phi_{\mathrm{target}}|$')
    axes[2].set_xlabel(r'$S$')
    axes[2].set_ylabel(r'$t$')
    axes[2].legend(loc='upper right', fontsize=9)
    add_colorbar(fig, axes[2], im2, label='Absolute diff')

    fig.suptitle(r'Figure 2: Phase-Field $\phi$ Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig2_phi_analysis.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "fig2_phi_analysis.png"))
    plt.close(fig)
    print("[Fig 2] φ analysis saved.")


# ============================================================
# 图 3: 自由边界提取对比 (φ=0.5 等值线 vs FDM)
# ============================================================
def plot_fig3_free_boundary(data):
    """从 φ 中提取 0.5 等值线，与 FDM 自由边界对比"""
    S = data["S"]
    t = data["t"]
    phi = data["phi"]
    fb = data["fdm_boundary"]

    if len(fb) != len(t):
        t_fb = np.linspace(0, t[-1], len(fb))
        fb_interp = interp1d(t_fb, fb, kind='linear', fill_value='extrapolate')(t)
    else:
        fb_interp = fb

    # 从 φ 提取自由边界
    fb_phi = np.full(len(t), np.nan)  # 用 nan 而非 0 标记无效点
    for i in range(len(t)):
        phi_row = phi[i, :]
        crossings = np.where(np.diff(np.sign(phi_row - 0.5)))[0]
        if len(crossings) > 0:
            idx = crossings[-1]
            if idx + 1 < len(S):
                s0, s1 = S[idx], S[idx + 1]
                p0, p1 = phi_row[idx], phi_row[idx + 1]
                denom = p1 - p0
                if abs(denom) > 1e-12:
                    fb_phi[i] = s0 + (0.5 - p0) / denom * (s1 - s0)
                else:
                    fb_phi[i] = 0.5 * (s0 + s1)

    # 对端点做特殊处理: 如果 nan, 用 FDM 值兜底
    for i in range(len(t)):
        if np.isnan(fb_phi[i]):
            fb_phi[i] = fb_interp[i]

    # 只在有效区间统计误差 (排除 t 非常接近 0 和 T 的端点)
    t_margin = 0.02 * t[-1]
    valid = (t > t_margin) & (t < t[-1] - t_margin) & (~np.isnan(fb_phi))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t, fb_interp, 'b-', linewidth=2.0,
            label=r'FDM free boundary $S_f^{\mathrm{FDM}}(t)$')
    ax.plot(t, fb_phi, 'r--', linewidth=2.0,
            label=r'$\phi = 0.5$ contour $S_f^{\phi}(t)$')
    ax.fill_between(t, 0, np.minimum(fb_interp, fb_phi),
                     alpha=0.15, color='gray', label='Exercise region')
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'Stock price $S$')
    ax.set_title('Figure 3: Free Boundary Extraction')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3)

    if valid.sum() > 0:
        fb_err = np.abs(fb_phi[valid] - fb_interp[valid])
        mean_fb = np.mean(np.abs(fb_interp[valid]))
        ax.text(0.98, 0.05,
                f'Mean |err| = {fb_err.mean():.2f}\n'
                f'Max |err| = {fb_err.max():.2f}\n'
                f'Rel err = {fb_err.mean() / (mean_fb + 1e-8):.4f}',
                transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig3_free_boundary.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "fig3_free_boundary.png"))
    plt.close(fig)
    print("[Fig 3] Free boundary comparison saved.")


# ============================================================
# 图 4: 训练损失曲线 (Stage 1 + Stage 2 各分量)
# ============================================================
def plot_fig4_training_loss(history):
    """多面板损失曲线"""
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.45)  # 加大间距

    # --- Stage 1: MSE + RelErr ---
    ax0 = fig.add_subplot(gs[0, 0])
    s1 = history.get("stage1", {})
    if "mse" in s1 and len(s1["mse"]) > 0:
        epochs_s1 = np.arange(1, len(s1["mse"]) + 1)
        ax0.semilogy(epochs_s1, s1["mse"], 'b-', linewidth=1.5, label='MSE (norm)')
        ax0.set_xlabel('Epoch')
        ax0.set_ylabel('Loss')
        ax0.set_title('Stage 1: MLP Pre-training')
        ax0.legend(loc='upper left')
        ax0.grid(True, alpha=0.3)

        ax0r = ax0.twinx()
        if "rel_err" in s1:
            ax0r.plot(epochs_s1, s1["rel_err"], 'r--', linewidth=1.2, label='RelErr')
            ax0r.set_ylabel('Relative Error', color='r', labelpad=10)
            ax0r.tick_params(axis='y', labelcolor='r')
            ax0r.legend(loc='center right')

    # --- Stage 1.5: BCE ---
    ax1 = fig.add_subplot(gs[0, 1])
    s15 = history.get("stage1_5", history.get("stage15", {}))
    bce_key = None
    for candidate in ["bce", "stage15_bce_list", "loss", "warm_mse", "BCE"]:
        if candidate in s15 and len(s15[candidate]) > 0:
            bce_key = candidate
            break
    if bce_key is not None:
        bce_data = s15[bce_key]
        log_interval = s15.get("log_interval", 300)
        epochs_s15 = np.arange(1, len(bce_data) + 1) * log_interval
        ax1.plot(epochs_s15, bce_data, 'g-o', linewidth=1.5,
                 markersize=4, label='BCE')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('BCE Loss', labelpad=10)
        ax1.set_title(r'Stage 1.5: $\phi$ Warm-Start')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # y 轴范围微调，避免刻度太密
        ymin, ymax = min(bce_data), max(bce_data)
        margin = (ymax - ymin) * 0.15 if ymax > ymin else 0.001
        ax1.set_ylim(ymin - margin, ymax + margin)
    else:
        ax1.text(0.5, 0.5, 'No Stage 1.5 data',
                 transform=ax1.transAxes, ha='center', va='center',
                 fontsize=12, color='gray')
        ax1.set_title(r'Stage 1.5: $\phi$ Warm-Start')

    # --- Stage 2: PDE / Exercise / BC ---
    s2 = history.get("stage2", {})

    ax2 = fig.add_subplot(gs[0, 2])
    if "pde" in s2 and len(s2["pde"]) > 0:
        epochs_s2 = np.arange(1, len(s2["pde"]) + 1)
        ax2.semilogy(epochs_s2, s2["pde"], 'b-', linewidth=1.2, label='PDE')
        if "exercise" in s2:
            ax2.semilogy(epochs_s2, s2["exercise"], 'r-', linewidth=1.2, label='Exercise')
        if "bc" in s2:
            ax2.semilogy(epochs_s2, s2["bc"], 'g-', linewidth=1.2, label='BC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Stage 2: PDE / Exercise / BC')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    # --- Stage 2: Interface Energy ---
    ax3 = fig.add_subplot(gs[1, 0])
    if "interface" in s2 and len(s2["interface"]) > 0:
        epochs_s2 = np.arange(1, len(s2["interface"]) + 1)
        ax3.plot(epochs_s2, s2["interface"], 'm-', linewidth=1.5, label='Interface')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title(r'Stage 2: Interface Energy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # --- Stage 2: Anchor Loss ---
    ax4 = fig.add_subplot(gs[1, 1])
    if "anchor" in s2 and len(s2["anchor"]) > 0:
        epochs_s2 = np.arange(1, len(s2["anchor"]) + 1)
        ax4.semilogy(epochs_s2, s2["anchor"], 'c-', linewidth=1.5, label='Anchor')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Stage 2: Anchor Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # --- Stage 2: Total Loss & RelErr ---
    ax5 = fig.add_subplot(gs[1, 2])
    if "total" in s2 and len(s2["total"]) > 0:
        epochs_s2 = np.arange(1, len(s2["total"]) + 1)
        ax5.semilogy(epochs_s2, s2["total"], 'k-', linewidth=1.5, label='Total Loss')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss')
        ax5.set_title('Stage 2: Total Loss & RelErr')
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.3)

        ax5r = ax5.twinx()
        if "rel_err" in s2:
            ax5r.semilogy(epochs_s2, s2["rel_err"], 'r--', linewidth=1.2, label='RelErr')
            ax5r.set_ylabel('Relative Error', color='r', labelpad=10)
            ax5r.tick_params(axis='y', labelcolor='r')
            ax5r.legend(loc='upper right')

    fig.suptitle('Figure 4: Training Loss History', fontsize=14, y=1.01)
    fig.savefig(os.path.join(OUTPUT_DIR, "fig4_training_loss.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "fig4_training_loss.png"))
    plt.close(fig)
    print("[Fig 4] Training loss history saved.")



# ============================================================
# 图 5: 固定时间截面 V(S) 对比
# ============================================================
def plot_fig5_time_slices(data):
    """在几个代表性时刻 t 绘制 V(S) 截面对比"""
    S = data["S"]
    t = data["t"]
    V_pred = data["V_pred"]
    V_fdm = data["V_fdm"]
    fb = data["fdm_boundary"]
    K = 100.0  # strike

    if len(fb) != len(t):
        t_fb = np.linspace(0, t[-1], len(fb))
        fb_interp = interp1d(t_fb, fb, kind='linear', fill_value='extrapolate')(t)
    else:
        fb_interp = fb

    # 选择时间截面
    time_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左：V(S) 截面
    ax = axes[0]
    for i, tf in enumerate(time_fracs):
        idx = np.argmin(np.abs(t - tf * t[-1]))
        t_val = t[idx]
        ax.plot(S, V_fdm[idx, :], '-', color=colors[i], linewidth=1.8,
                label=f'FDM, $t={t_val:.2f}$')
        ax.plot(S, V_pred[idx, :], '--', color=colors[i], linewidth=1.5,
                label=f'PINN, $t={t_val:.2f}$', alpha=0.8)

    payoff = np.maximum(K - S, 0)
    ax.plot(S, payoff, 'k:', linewidth=1.2, label=r'Payoff $\max(K-S, 0)$')
    ax.set_xlabel(r'Stock price $S$')
    ax.set_ylabel(r'Option value $V$')
    ax.set_title('(a) Value function at selected times')
    ax.set_xlim(0, 200)
    ax.set_ylim(-2, K + 5)
    ax.legend(fontsize=7.5, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 右：误差截面
    ax2 = axes[1]
    for i, tf in enumerate(time_fracs):
        idx = np.argmin(np.abs(t - tf * t[-1]))
        t_val = t[idx]
        err = V_pred[idx, :] - V_fdm[idx, :]
        ax2.plot(S, err, '-', color=colors[i], linewidth=1.5,
                 label=f'$t={t_val:.2f}$')
        # 标记自由边界位置
        ax2.axvline(x=fb_interp[idx], color=colors[i], linestyle=':', alpha=0.5)

    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.set_xlabel(r'Stock price $S$')
    ax2.set_ylabel(r'$V_{\mathrm{PINN}} - V_{\mathrm{FDM}}$')
    ax2.set_title('(b) Pricing error at selected times')
    ax2.set_xlim(0, 200)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Figure 5: Time-Slice Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig5_time_slices.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "fig5_time_slices.png"))
    plt.close(fig)
    print("[Fig 5] Time slices saved.")


# ============================================================
# 图 6: φ 截面 + 直方图
# ============================================================
def plot_fig6_phi_details(data):
    """左：几个时刻的 φ(S) 截面；右：φ 值的直方图"""
    S = data["S"]
    t = data["t"]
    phi = data["phi"]
    fb = data["fdm_boundary"]
    K = 100.0

    if len(fb) != len(t):
        t_fb = np.linspace(0, t[-1], len(fb))
        fb_interp = interp1d(t_fb, fb, kind='linear', fill_value='extrapolate')(t)
    else:
        fb_interp = fb

    time_fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左：φ(S) 截面
    ax = axes[0]
    for i, tf in enumerate(time_fracs):
        idx = np.argmin(np.abs(t - tf * t[-1]))
        t_val = t[idx]
        ax.plot(S, phi[idx, :], '-', color=colors[i], linewidth=1.8,
                label=f'$t={t_val:.3f}$')
        ax.axvline(x=fb_interp[idx], color=colors[i], linestyle=':', alpha=0.6)

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r'Stock price $S$')
    ax.set_ylabel(r'$\phi(S, t)$')
    ax.set_title(r'(a) $\phi$ profiles at selected times')
    ax.set_xlim(0, 150)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(20, 0.15, 'Exercise\nregion', fontsize=10, ha='center', color='gray',
            style='italic')
    ax.text(120, 0.85, 'Continuation\nregion', fontsize=10, ha='center', color='gray',
            style='italic')

    # 右：直方图
    ax2 = axes[1]
    phi_flat = phi.flatten()
    ax2.hist(phi_flat, bins=100, density=True, color='steelblue', alpha=0.7,
             edgecolor='white', linewidth=0.3)
    ax2.set_xlabel(r'$\phi$ value')
    ax2.set_ylabel('Density')
    ax2.set_title(r'(b) Distribution of $\phi$ values')
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=1.2, label=r'$\phi=0.5$')

    # 统计信息
    ex_frac = np.mean(phi_flat < 0.1)
    cont_frac = np.mean(phi_flat > 0.9)
    trans_frac = np.mean((phi_flat >= 0.1) & (phi_flat <= 0.9))
    ax2.text(0.97, 0.95,
             f'Exercise ($\\phi<0.1$): {ex_frac:.1%}\n'
             f'Transition: {trans_frac:.1%}\n'
             f'Continuation ($\\phi>0.9$): {cont_frac:.1%}\n'
             f'Mean: {phi_flat.mean():.4f}\n'
             f'Std: {phi_flat.std():.4f}',
             transform=ax2.transAxes, fontsize=9, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(r'Figure 6: Phase-Field $\phi$ Details', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig6_phi_details.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "fig6_phi_details.png"))
    plt.close(fig)
    print("[Fig 6] φ details saved.")


# ============================================================
# 图 7: 3D 曲面图 V(S,t)
# ============================================================
def plot_fig7_3d_surface(data):
    """V_pred 的 3D 曲面 + 自由边界投影"""
    S = data["S"]
    t = data["t"]
    V_pred = data["V_pred"]
    fb = data["fdm_boundary"]

    if len(fb) != len(t):
        t_fb = np.linspace(0, t[-1], len(fb))
        fb_interp = interp1d(t_fb, fb, kind='linear', fill_value='extrapolate')(t)
    else:
        fb_interp = fb

    SS, TT = np.meshgrid(S, t)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 降采样以提高绘图速度
    step_s = max(1, len(S) // 80)
    step_t = max(1, len(t) // 50)
    ax.plot_surface(SS[::step_t, ::step_s], TT[::step_t, ::step_s],
                    V_pred[::step_t, ::step_s],
                    cmap='RdYlBu_r', alpha=0.85, linewidth=0, antialiased=True)

    # 自由边界线
    fb_V = np.maximum(100.0 - fb_interp, 0)
    ax.plot(fb_interp, t, fb_V, 'k-', linewidth=2.5, label='Free boundary', zorder=10)

    ax.set_xlabel(r'$S$', labelpad=10)
    ax.set_ylabel(r'$t$', labelpad=10)
    ax.set_zlabel(r'$V(S,t)$', labelpad=10)
    ax.set_title('Figure 7: 3D Surface of PINN Solution', pad=15)
    ax.view_init(elev=25, azim=-60)
    ax.legend(loc='upper left')

    fig.savefig(os.path.join(OUTPUT_DIR, "fig7_3d_surface.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "fig7_3d_surface.png"))
    plt.close(fig)
    print("[Fig 7] 3D surface saved.")


# ============================================================
# 图 8: 误差空间分布热力图 (相对误差)
# ============================================================
def plot_fig8_relative_error_map(data):
    """逐点相对误差热力图"""
    S = data["S"]
    t = data["t"]
    V_pred = data["V_pred"]
    V_fdm = data["V_fdm"]
    fb = data["fdm_boundary"]

    if len(fb) != len(t):
        t_fb = np.linspace(0, t[-1], len(fb))
        fb_interp = interp1d(t_fb, fb, kind='linear', fill_value='extrapolate')(t)
    else:
        fb_interp = fb

    # 逐点相对误差（避免分母为零）
    denom = np.maximum(np.abs(V_fdm), 1e-6)
    rel_err_map = np.abs(V_pred - V_fdm) / denom
    # 对深虚值区域（V_fdm ≈ 0）cap 相对误差
    mask_small = V_fdm < 0.01
    rel_err_map[mask_small] = np.nan

    SS, TT = np.meshgrid(S, t)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(SS, TT, rel_err_map * 100, cmap='YlOrRd', shading='auto',
                        vmin=0, vmax=np.nanpercentile(rel_err_map * 100, 95))
    ax.plot(fb_interp, t, 'k-', linewidth=2.0, label='Free boundary (FDM)')
    ax.set_xlabel(r'$S$')
    ax.set_ylabel(r'$t$')
    ax.set_title('Figure 8: Pointwise Relative Error (%)')
    ax.legend(loc='upper right')
    add_colorbar(fig, ax, im, label='Relative error (%)')

    # 统计
    valid = ~np.isnan(rel_err_map)
    ax.text(0.02, 0.02,
            f'Mean: {np.nanmean(rel_err_map)*100:.4f}%\n'
            f'Median: {np.nanmedian(rel_err_map)*100:.4f}%\n'
            f'95th: {np.nanpercentile(rel_err_map*100, 95):.4f}%',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig8_relative_error_map.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "fig8_relative_error_map.png"))
    plt.close(fig)
    print("[Fig 8] Relative error map saved.")


# ============================================================
# 图 9: φ 与 V-Payoff 联合展示
# ============================================================
def plot_fig9_phi_v_payoff(data):
    """展示 φ 如何与 V-Payoff 的关系吻合"""
    S = data["S"]
    t = data["t"]
    V_pred = data["V_pred"]
    phi = data["phi"]
    K = 100.0

    payoff = np.maximum(K - S[np.newaxis, :], 0)
    v_minus_payoff = V_pred - payoff

    SS, TT = np.meshgrid(S, t)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：V - Payoff
    vmin_val = np.percentile(v_minus_payoff, 1)
    vmax_val = np.percentile(v_minus_payoff, 99)
    # 确保 vcenter=0 在范围内
    if vmin_val >= 0:
        vmin_val = -0.1
    if vmax_val <= 0:
        vmax_val = 0.1
    norm = TwoSlopeNorm(vmin=vmin_val, vcenter=0, vmax=vmax_val)
    im0 = axes[0].pcolormesh(SS, TT, v_minus_payoff, cmap='RdBu_r',
                              shading='auto', norm=norm)
    axes[0].set_title(r'$V_{\mathrm{PINN}} - \Psi$ (early exercise premium)')
    axes[0].set_xlabel(r'$S$')
    axes[0].set_ylabel(r'$t$')
    add_colorbar(fig, axes[0], im0, label=r'$V - \Psi$')

    # 右：φ 叠加 V-Payoff = 0 的等值线 (修正: 控制标签数量)
    im1 = axes[1].pcolormesh(SS, TT, phi, cmap='coolwarm', shading='auto',
                              vmin=0, vmax=1)
    cs = axes[1].contour(SS, TT, v_minus_payoff, levels=[0],
                          colors='lime', linewidths=2.5)
    # 控制标签数量: 只标注 1-2 个
    try:
        labels = axes[1].clabel(cs, fmt=r'$V=\Psi$', fontsize=10,
                                 inline=True, inline_spacing=50)
        # 只保留前 2 个标签
        if labels is not None and len(labels) > 2:
            for lbl in labels[2:]:
                lbl.remove()
    except Exception:
        # 如果 clabel 失败（无等值线），手动添加文字标注
        axes[1].text(80, 0.5, r'$V=\Psi$', fontsize=11, color='lime',
                      fontweight='bold', ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.2',
                                facecolor='black', alpha=0.5))
    axes[1].set_title(r'$\phi$ with $V=\Psi$ contour')
    axes[1].set_xlabel(r'$S$')
    axes[1].set_ylabel(r'$t$')
    add_colorbar(fig, axes[1], im1, label=r'$\phi$')

    fig.suptitle('Figure 9: Phase-Field vs Early Exercise Premium',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig9_phi_v_payoff.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, "fig9_phi_v_payoff.png"))
    plt.close(fig)
    print("[Fig 9] φ vs V-Payoff saved.")


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 60)
    print("加载数据...")
    data, history = load_data()

    # 打印基本统计
    print(f"  S shape: {data['S'].shape}, range [{data['S'].min():.1f}, {data['S'].max():.1f}]")
    print(f"  t shape: {data['t'].shape}, range [{data['t'].min():.4f}, {data['t'].max():.4f}]")
    print(f"  V_pred shape: {data['V_pred'].shape}")
    print(f"  V_fdm shape: {data['V_fdm'].shape}")
    print(f"  phi shape: {data['phi'].shape}")
    print(f"  phi_target shape: {data['phi_target'].shape}")

    V_pred = data['V_pred']
    V_fdm = data['V_fdm']
    rel_err = np.sqrt(np.mean((V_pred - V_fdm)**2)) / (np.sqrt(np.mean(V_fdm**2)) + 1e-12)
    print(f"  Overall V RelErr: {rel_err:.6f} ({rel_err*100:.4f}%)")

    phi = data['phi']
    print(f"  φ stats: mean={phi.mean():.4f}, std={phi.std():.4f}, "
          f"min={phi.min():.4f}, max={phi.max():.4f}")
    print(f"  φ<0.1: {np.mean(phi<0.1):.1%}, φ>0.9: {np.mean(phi>0.9):.1%}")
    print("=" * 60)

    # 逐图生成
    print("\n生成图表...")
    plot_fig1_value_comparison(data)
    plot_fig2_phi_analysis(data)
    plot_fig3_free_boundary(data)
    plot_fig4_training_loss(history)
    plot_fig5_time_slices(data)
    plot_fig6_phi_details(data)
    plot_fig7_3d_surface(data)
    plot_fig8_relative_error_map(data)
    plot_fig9_phi_v_payoff(data)

    print("\n" + "=" * 60)
    print(f"全部 9 张图已保存至 ./{OUTPUT_DIR}/")
    print("PDF 版本可直接用于 LaTeX 论文插入。")
    print("=" * 60)


if __name__ == "__main__":
    main()
