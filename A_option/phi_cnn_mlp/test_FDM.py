"""
fdm_convergence_test.py
FDM 网格收敛性测试 — 用于 Appendix A (Table A.1)

测试 5 级网格，以最细网格为参考解，计算逐级误差 n和收敛阶。
"""

import numpy as np
import time


# ============================================================
# Thomas 算法 (与 s1_cnn_phasefield.py 完全一致)
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


# ============================================================
# FDM 求解器 (与 s1_cnn_phasefield.py 完全一致)
# ============================================================
def generate_fdm_american_put(S_max, T, K, r, sigma, Ns, Nt):
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
# 期权参数 (与 Config 一致)
# ============================================================
S_max = 200.0
T     = 1.0
K     = 100.0
r     = 0.05
sigma = 0.2

# ============================================================
# 5 级网格: 每级 Ns 和 Nt 同时翻倍
# ============================================================
grid_levels = [
    {"label": "Level 1", "Ns": 125,  "Nt": 500},
    {"label": "Level 2", "Ns": 250,  "Nt": 1000},
    {"label": "Level 3", "Ns": 500,  "Nt": 2000},   # <-- 论文中使用的网格
    {"label": "Level 4", "Ns": 1000, "Nt": 4000},
    {"label": "Level 5", "Ns": 2000, "Nt": 8000},   # 最细网格 (参考解)
]

# ============================================================
# 逐级求解
# ============================================================
solutions = []
print("=" * 80)
print("FDM Grid Convergence Test")
print("=" * 80)
print(f"Parameters: K={K}, r={r}, sigma={sigma}, T={T}, S_max={S_max}")
print("-" * 80)

for level in grid_levels:
    Ns, Nt = level["Ns"], level["Nt"]
    dS = S_max / Ns
    dt = T / Nt

    print(f"\n{level['label']}: Ns={Ns}, Nt={Nt}, dS={dS:.4f}, dt={dt:.6f}, "
          f"grid shape=({Nt+1}, {Ns+1})")

    t0 = time.time()
    S, t_grid, V_all = generate_fdm_american_put(S_max, T, K, r, sigma, Ns, Nt)
    elapsed = time.time() - t0

    print(f"  Solved in {elapsed:.2f}s, V range: [{V_all.min():.6f}, {V_all.max():.6f}]")

    # 记录几个关键点的值
    # V(S=K, t=0) — at-the-money, inception
    idx_K = np.argmin(np.abs(S - K))
    V_atm = V_all[0, idx_K]
    print(f"  V(S={S[idx_K]:.1f}, t=0) = {V_atm:.8f}")

    solutions.append({
        "label": level["label"],
        "Ns": Ns, "Nt": Nt,
        "dS": dS, "dt": dt,
        "S": S, "t": t_grid, "V": V_all,
        "time": elapsed,
        "V_atm": V_atm
    })

# ============================================================
# 用最细网格 (Level 5) 作为参考解，计算各级误差
# ============================================================
print("\n" + "=" * 80)
print("Convergence Analysis (reference = Level 5)")
print("=" * 80)

ref = solutions[-1]
S_ref, t_ref, V_ref = ref["S"], ref["t"], ref["V"]

from scipy.interpolate import RegularGridInterpolator

ref_interp = RegularGridInterpolator(
    (t_ref, S_ref), V_ref, method='linear', bounds_error=False, fill_value=None
)

print(f"\n{'Level':<10} {'Ns':<8} {'Nt':<8} {'dS':<10} {'dt':<12} "
      f"{'L_inf err':<14} {'L2 err':<14} {'V(K,0)':<14} {'|V-V_ref|':<14} {'Time(s)':<10}")
print("-" * 120)

errors_Linf = []
errors_L2 = []

for sol in solutions:
    S_k, t_k, V_k = sol["S"], sol["t"], sol["V"]

    # 将参考解插值到当前网格上
    tt_k, ss_k = np.meshgrid(t_k, S_k, indexing='ij')
    points = np.stack([tt_k.ravel(), ss_k.ravel()], axis=-1)
    V_ref_on_k = ref_interp(points).reshape(V_k.shape)

    # 误差
    diff = np.abs(V_k - V_ref_on_k)
    Linf = diff.max()
    L2 = np.sqrt(np.mean(diff ** 2))

    # ATM 点误差
    idx_K = np.argmin(np.abs(S_k - K))
    V_atm_err = abs(sol["V_atm"] - ref["V_atm"])

    errors_Linf.append(Linf)
    errors_L2.append(L2)

    print(f"{sol['label']:<10} {sol['Ns']:<8} {sol['Nt']:<8} "
          f"{sol['dS']:<10.4f} {sol['dt']:<12.6f} "
          f"{Linf:<14.6e} {L2:<14.6e} {sol['V_atm']:<14.8f} {V_atm_err:<14.6e} "
          f"{sol['time']:<10.2f}")

# ============================================================
# 收敛阶
# ============================================================
print("\n" + "=" * 80)
print("Convergence Rates (between consecutive levels)")
print("=" * 80)
print(f"\n{'Levels':<20} {'L_inf ratio':<16} {'L_inf order':<16} "
      f"{'L2 ratio':<16} {'L2 order':<16}")
print("-" * 84)

for i in range(1, len(solutions) - 1):  # 不包括参考解自身
    if errors_Linf[i] > 1e-15 and errors_Linf[i-1] > 1e-15:
        ratio_Linf = errors_Linf[i - 1] / errors_Linf[i]
        order_Linf = np.log2(ratio_Linf)
    else:
        ratio_Linf = float('inf')
        order_Linf = float('nan')

    if errors_L2[i] > 1e-15 and errors_L2[i-1] > 1e-15:
        ratio_L2 = errors_L2[i - 1] / errors_L2[i]
        order_L2 = np.log2(ratio_L2)
    else:
        ratio_L2 = float('inf')
        order_L2 = float('nan')

    print(f"{solutions[i-1]['label']}→{solutions[i]['label']:<10} "
          f"{ratio_Linf:<16.4f} {order_Linf:<16.4f} "
          f"{ratio_L2:<16.4f} {order_L2:<16.4f}")

# ============================================================
# 额外: 逐级差 (相邻网格差异，不依赖参考解)
# ============================================================
print("\n" + "=" * 80)
print("Successive Differences (Level k vs Level k+1)")
print("=" * 80)
print(f"\n{'Levels':<20} {'L_inf diff':<16} {'Ratio':<16} {'Order':<16}")
print("-" * 68)

successive_diffs = []
for i in range(len(solutions) - 1):
    sol_coarse = solutions[i]
    sol_fine = solutions[i + 1]

    # 将细网格插值到粗网格
    fine_interp = RegularGridInterpolator(
        (sol_fine["t"], sol_fine["S"]), sol_fine["V"],
        method='linear', bounds_error=False, fill_value=None
    )
    tt_c, ss_c = np.meshgrid(sol_coarse["t"], sol_coarse["S"], indexing='ij')
    pts = np.stack([tt_c.ravel(), ss_c.ravel()], axis=-1)
    V_fine_on_coarse = fine_interp(pts).reshape(sol_coarse["V"].shape)

    diff_Linf = np.abs(sol_coarse["V"] - V_fine_on_coarse).max()
    successive_diffs.append(diff_Linf)

for i, d in enumerate(successive_diffs):
    if i > 0 and successive_diffs[i] > 1e-15:
        ratio = successive_diffs[i - 1] / successive_diffs[i]
        order = np.log2(ratio)
    else:
        ratio = float('nan')
        order = float('nan')

    label = f"{solutions[i]['label']}→{solutions[i+1]['label']}"
    print(f"{label:<20} {d:<16.6e} {ratio:<16.4f} {order:<16.4f}")

# ============================================================
# 总结 (直接可填入论文 Table A.1)
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY FOR PAPER (Table A.1)")
print("=" * 80)
print(f"\n{'Grid (Nt+1)x(Ns+1)':<22} {'dS':<10} {'dt':<14} "
      f"{'||V-V_ref||_inf':<18} {'||V-V_ref||_2':<18} {'Rate (L_inf)':<14} {'Time(s)':<10}")
print("-" * 106)

for i, sol in enumerate(solutions):
    grid_str = f"({sol['Nt']+1})x({sol['Ns']+1})"

    if i == len(solutions) - 1:
        err_str_inf = "--- (reference)"
        err_str_l2 = "--- (reference)"
        rate_str = "---"
    else:
        err_str_inf = f"{errors_Linf[i]:.6e}"
        err_str_l2 = f"{errors_L2[i]:.6e}"
        if i > 0 and errors_Linf[i] > 1e-15:
            rate = np.log2(errors_Linf[i-1] / errors_Linf[i])
            rate_str = f"{rate:.2f}"
        else:
            rate_str = "---"

    print(f"{grid_str:<22} {sol['dS']:<10.4f} {sol['dt']:<14.6f} "
          f"{err_str_inf:<18} {err_str_l2:<18} {rate_str:<14} {sol['time']:<10.2f}")

print("\nDone! Copy the SUMMARY table into Appendix A.")
