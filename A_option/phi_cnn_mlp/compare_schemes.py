# ================================================================
# compare_schemes.py
# 合并比较三个方案的结果
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import json, os

# FDM参考解
from scheme1_cnn_phasefield import Config, generate_fdm_solution

plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

config = Config()
S_fdm, t_fdm, V_fdm_full = generate_fdm_solution(config)


def load_scheme(name):
    data = np.load(f'{name}_results.npz')
    with open(f'{name}_history.json', 'r') as f:
        hist = json.load(f)
    return data, hist


def compute_free_boundary_fdm():
    """从FDM解提取行权边界"""
    payoff = np.maximum(config.K - S_fdm, 0)
    fb = np.zeros(len(t_fdm))
    for j in range(len(t_fdm)):
        diff = np.abs(V_fdm_full[:, j] - payoff)
        exercise = np.where(diff < 0.5)[0]
        if len(exercise) > 0:
            fb[j] = S_fdm[exercise[-1]]
        else:
            fb[j] = 0
    return fb


def compute_metrics(V_pred, S_test, t_test):
    """计算关键指标"""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((S_fdm, t_fdm), V_fdm_full, method='linear')
    Sm, tm = np.meshgrid(S_test, t_test, indexing='ij')
    pts = np.stack([Sm.ravel(), tm.ravel()], axis=-1)
    V_ref = interp(pts).reshape(V_pred.shape)

    # L2相对误差
    l2_rel = np.sqrt(np.mean((V_pred - V_ref)**2)) / (np.sqrt(np.mean(V_ref**2)) + 1e-10)

    # Linf误差
    linf = np.max(np.abs(V_pred - V_ref))

    # 无套利偏差
    payoff = np.maximum(config.K - Sm, 0)
    arb = np.max(np.maximum(payoff - V_pred, 0))

    return l2_rel, linf, arb, V_ref


# ================================================================
# 加载数据
# ================================================================
schemes = {}
scheme_names = {
    'scheme1': 'CNN-φ + MLP-V (Ours)',
    'scheme2': 'MLP-φ + MLP-V',
    'scheme3': 'PINN + Penalty',
}
colors = {'scheme1': '#E63946', 'scheme2': '#457B9D', 'scheme3': '#2A9D8F'}

for key in ['scheme1', 'scheme2', 'scheme3']:
    fname = f'{key}_results.npz'
    if os.path.exists(fname):
        schemes[key] = load_scheme(key)
        print(f"✓ 已加载 {key}")
    else:
        print(f"✗ 未找到 {fname}, 请先运行对应脚本")

if len(schemes) == 0:
    print("没有可用的结果，退出。")
    exit()

fb_fdm = compute_free_boundary_fdm()

# ================================================================
# 图1: V曲面对比 (FDM + 三个方案)
# ================================================================
n_panels = 1 + len(schemes)
fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
if n_panels == 1:
    axes = [axes]

# FDM参考
from scipy.interpolate import RegularGridInterpolator
S_plot = np.linspace(0, config.S_max, 200)
t_plot = np.linspace(0, config.T, 100)
interp_fdm = RegularGridInterpolator((S_fdm, t_fdm), V_fdm_full, method='linear')
Sm_p, tm_p = np.meshgrid(S_plot, t_plot, indexing='ij')
V_fdm_plot = interp_fdm(np.stack([Sm_p.ravel(), tm_p.ravel()], -1)).reshape(200, 100)
payoff_plot = np.maximum(config.K - Sm_p, 0)

im = axes[0].contourf(tm_p, Sm_p, V_fdm_plot, levels=50, cmap='viridis')
axes[0].contour(tm_p, Sm_p, V_fdm_plot - payoff_plot, levels=[0], colors='red', linewidths=2)
axes[0].set_title('FDM Reference', fontweight='bold')
axes[0].set_xlabel('t')
axes[0].set_ylabel('S')
fig.colorbar(im, ax=axes[0], shrink=0.8)

for idx, (key, (data, hist)) in enumerate(schemes.items()):
    ax = axes[idx + 1]
    S, t, V = data['S'], data['t'], data['V']
    Sm, tm = np.meshgrid(S, t, indexing='ij')
    payoff = np.maximum(config.K - Sm, 0)
    im = ax.contourf(tm, Sm, V, levels=50, cmap='viridis')
    if 'phi' in data:
        ax.contour(tm, Sm, data['phi'], levels=[0.5], colors='red', linewidths=2, linestyles='--')
    else:
        ax.contour(tm, Sm, V - payoff, levels=[0], colors='red', linewidths=2, linestyles='--')
    ax.set_title(scheme_names[key], fontweight='bold')
    ax.set_xlabel('t')
    ax.set_ylabel('S')
    fig.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig('compare_V_surfaces.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 保存: compare_V_surfaces.png")

# ================================================================
# 图2: φ分布对比 (方案1 vs 方案2, 方案3没有φ)
# ================================================================
phi_schemes = {k: v for k, v in schemes.items() if 'phi' in v[0]}
if len(phi_schemes) >= 1:
    fig, axes = plt.subplots(1, len(phi_schemes), figsize=(6 * len(phi_schemes), 5))
    if len(phi_schemes) == 1:
        axes = [axes]
    for idx, (key, (data, hist)) in enumerate(phi_schemes.items()):
        ax = axes[idx]
        S, t, phi = data['S'], data['t'], data['phi']
        Sm, tm = np.meshgrid(S, t, indexing='ij')
        im = ax.contourf(tm, Sm, phi, levels=50, cmap='coolwarm', vmin=0, vmax=1)
        ax.contour(tm, Sm, phi, levels=[0.5], colors='black', linewidths=2)
        # 叠加FDM行权边界
        ax.plot(t_fdm, fb_fdm, 'g--', linewidth=2, label='FDM boundary')
        ax.set_title(f'φ: {scheme_names[key]}', fontweight='bold')
        ax.set_xlabel('t')
        ax.set_ylabel('S')
        ax.legend(loc='upper right')
        fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig('compare_phi.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 保存: compare_phi.png")

# ================================================================
# 图3: 训练曲线对比
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# (a) 总损失/能量
ax = axes[0]
for key, (data, hist) in schemes.items():
    metric = hist.get('energy', hist.get('pde', hist['loss']))
    ax.semilogy(metric, color=colors[key], label=scheme_names[key], alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Energy / PDE Loss')
ax.set_title('(a) Training Loss', fontweight='bold')
ax.legend(fontsize=9)

# (b) 无套利偏差
ax = axes[1]
for key, (data, hist) in schemes.items():
    ax.semilogy(hist['arb'], color=colors[key], label=scheme_names[key], alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Max Arbitrage Violation')
ax.set_title('(b) No-Arbitrage Violation', fontweight='bold')
ax.legend(fontsize=9)

# (c) φ精度 (仅方案1,2)
ax = axes[2]
for key, (data, hist) in schemes.items():
    if 'phi_acc' in hist:
        ax.plot(hist['phi_acc'], color=colors[key], label=scheme_names[key], alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('φ Classification Accuracy')
ax.set_title('(c) Phase Field Accuracy', fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim([0.5, 1.02])

plt.tight_layout()
plt.savefig('compare_training.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 保存: compare_training.png")

# ================================================================
# 图4: t=0时刻V(S)切面对比
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) V(S, t=0) 对比
ax = axes[0]
# FDM参考
ax.plot(S_fdm, V_fdm_full[:, 0], 'k-', linewidth=2.5, label='FDM Reference')
ax.plot(S_fdm, np.maximum(config.K - S_fdm, 0), 'k--', linewidth=1, alpha=0.5, label='Payoff')
for key, (data, hist) in schemes.items():
    ax.plot(data['S'], data['V'][:, 0], color=colors[key], linewidth=1.5,
            linestyle='--', label=scheme_names[key])
ax.set_xlim([0, 200])
ax.set_xlabel('S')
ax.set_ylabel('V(S, t=0)')
ax.set_title('(a) Option Value at t=0', fontweight='bold')
ax.legend(fontsize=9)

# (b) 误差 |V_pred - V_ref| at t=0
ax = axes[1]
for key, (data, hist) in schemes.items():
    S_test, t_test, V_pred = data['S'], data['t'], data['V']
    l2, linf, arb, V_ref = compute_metrics(V_pred, S_test, t_test)
    err = np.abs(V_pred[:, 0] - V_ref[:, 0])
    ax.semilogy(S_test, err + 1e-10, color=colors[key], linewidth=1.5, label=f'{scheme_names[key]} (L²={l2:.4f})')
ax.set_xlim([0, 200])
ax.set_xlabel('S')
ax.set_ylabel('|V_pred - V_ref|')
ax.set_title('(b) Absolute Error at t=0', fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('compare_t0_slice.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 保存: compare_t0_slice.png")

# ================================================================
# 图5: 行权边界对比
# ================================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_fdm, fb_fdm, 'k-', linewidth=2.5, label='FDM Reference')

for key, (data, hist) in schemes.items():
    S, t = data['S'], data['t']
    if 'phi' in data:
        phi = data['phi']
        Sm, tm = np.meshgrid(S, t, indexing='ij')
        # φ=0.5等值线
        fb_pred = np.zeros(len(t))
        for j in range(len(t)):
            cross = np.where(np.diff(np.sign(phi[:, j] - 0.5)))[0]
            if len(cross) > 0:
                i0 = cross[-1]
                # 线性插值
                frac = (0.5 - phi[i0, j]) / (phi[i0 + 1, j] - phi[i0, j] + 1e-10)
                fb_pred[j] = S[i0] + frac * (S[i0 + 1] - S[i0])
            else:
                fb_pred[j] = 0
        ax.plot(t, fb_pred, color=colors[key], linewidth=1.5, linestyle='--',
                label=f'{scheme_names[key]} (φ=0.5)')
    else:
        V = data['V']
        payoff = np.maximum(config.K - np.array(S)[:, None], 0)
        fb_pred = np.zeros(len(t))
        for j in range(len(t)):
            diff = np.abs(V[:, j] - payoff[:, j])
            ex = np.where(diff < 0.5)[0]
            fb_pred[j] = S[ex[-1]] if len(ex) > 0 else 0
        ax.plot(t, fb_pred, color=colors[key], linewidth=1.5, linestyle='--',
                label=scheme_names[key])

ax.set_xlabel('t')
ax.set_ylabel('S* (Exercise Boundary)')
ax.set_title('Exercise Boundary Comparison', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('compare_boundary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 保存: compare_boundary.png")

# ================================================================
# 汇总表
# ================================================================
print("\n" + "=" * 70)
print("定量比较汇总")
print("=" * 70)
print(f"{'方案':<30s} {'L²相对误差':<15s} {'L∞误差':<12s} {'Max Arb':<12s}")
print("-" * 70)

for key, (data, hist) in schemes.items():
    S, t, V = data['S'], data['t'], data['V']
    l2, linf, arb, _ = compute_metrics(V, S, t)
    print(f"{scheme_names[key]:<30s} {l2:<15.6f} {linf:<12.4f} {arb:<12.4f}")

print("=" * 70)
print("\n所有图表已保存。完成！")
