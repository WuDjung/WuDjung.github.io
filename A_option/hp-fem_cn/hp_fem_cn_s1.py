# hp_fem_cn_ssn.py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time
#=========一些其它配置===========

# 配置中文字体（使用系统中的“文泉驿微米黑”），解决负号“-”显示为方框的问题
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  
plt.rcParams['axes.unicode_minus'] = False  

# 记录开始时间
start_total = time.time()

# ========== 参数 ==========
K = 100.0; T = 1.0; r = 0.05; sigma = 0.2
x_max = 3.0          # 3K
M = 10               # 初始单元数（可再加密）
N = 500             # 时间步
p = 2                # 单元阶数（全局先固定）
# ==========================

dt = T / N
dx = x_max / M
nodes = np.linspace(0, x_max, M+1)   # 节点坐标

# ========== 局部基函数（一维 hierarchical） ==========
def local_matrices(xl, xr, p):
    h = xr - xl
    # 高斯点 & 权重（p+1 点即可精确 2p 阶）
    gp, gw = np.polynomial.legendre.leggauss(p+1)
    gp = 0.5*(gp + 1)*h + xl      # 映射到单元
    gw *= 0.5*h                   # 权重缩放

    n_loc = p + 1                 # 局部自由度
    Ke = np.zeros((n_loc, n_loc))
    Me = np.zeros_like(Ke)

    for xi, wi in zip(gp, gw):
        # 线性映射 x(ξ)
        x_val = xi
        # 基函数及其导数（hierarchical 模式）
        # 这里用 Legendre 模板的 hierarchical 基
        phi = np.array([1.0, xi - 0.5*(xl+xr)])
        if p >= 2:
            L2 = 0.5*(3*(xi**2) - 1)
            phi = np.append(phi, L2)
        if p >= 3:
            L3 = 0.5*(5*(xi**3) - 3*xi)
            phi = np.append(phi, L3)
        # 导数
        dphi = np.array([0.0, 1.0])
        if p >= 2:
            dphi = np.append(dphi, 3*xi)
        if p >= 3:
            dphi = np.append(dphi, 0.5*(15*xi**2 - 3))

        # 局部矩阵装配
        Ke += 0.5*sigma**2*x_val**2 * np.outer(dphi, dphi) * wi
        Ke += r*x_val * np.outer(dphi, phi) * wi
        Ke -= r * np.outer(phi, phi) * wi
        Me += np.outer(phi, phi) * wi
    return Ke, Me

# ========== 全局装配 ==========
def assemble_global(p):
    dof_map = []               # 单元 → 全局自由度
    tot_dof = 0
    # 使用全局变量M，循环遍历所有单元
    for e in range(M):
        dof_map.append(np.arange(tot_dof, tot_dof + p + 1))
        tot_dof += p + 1
    # 将局部变量重命名，避免与全局变量冲突
    K_mat = csr_matrix((tot_dof, tot_dof))
    M_mat = csr_matrix((tot_dof, tot_dof))
    # 单元循环
    for e in range(M):
        xl, xr = nodes[e], nodes[e+1]
        Ke, Me = local_matrices(xl, xr, p)
        loc = dof_map[e]
        K_mat[np.ix_(loc, loc)] += Ke
        M_mat[np.ix_(loc, loc)] += Me
    return K_mat, M_mat, dof_map, tot_dof

K_mat, M_mat, dof_map, n_dof = assemble_global(p)

# ========== CN 左右矩阵 ==========
A = M_mat - 0.5*dt*K_mat
B = M_mat + 0.5*dt*K_mat

# ========== 初始条件 ==========
x_nodal = np.linspace(0, x_max, n_dof)   # 这里简化：用单元节点线性插值
v0 = np.maximum(1.0 - x_nodal, 0.0)

# ========== 边界处理（直接消去） ==========
# 左端 x=0  v=1，右端 x=x_max v=0
bd_left = 0
bd_right = n_dof - 1
interior = np.setdiff1d(np.arange(n_dof), [bd_left, bd_right])
A_int = A[interior][:, interior]
B_int = B[interior][:, interior]

# ========== 时间推进：CN方法 ==========
v_cn = v0.copy()
for n in range(N):
    # 计算右端项，确保结果为一维数组
    rhs = B_int.dot(v_cn[interior]).ravel()
    # 边界贡献 - 确保形状匹配，使用toarray()转为密集数组并展平
    boundary_contribution = B[interior, bd_left].toarray().flatten() * 1.0
    rhs += boundary_contribution  # 现在两边都是一维数组，可正确广播
    # 解线性系统
    v_cn[interior] = spsolve(A_int, rhs)
    v_cn[bd_left], v_cn[bd_right] = 1.0, 0.0   # 强制边界

# ========== 时间推进：SSN 解 LCP ==========
# 定义收益函数（payoff），与初始条件保持一致
payoff = np.maximum(1.0 - x_nodal, 0.0)

u_ssn = v0.copy()
active = np.zeros_like(u_ssn, dtype=bool)

for n in range(N):
    b_rhs = (M_mat + 0.5*dt*K_mat)@u_ssn
    b_rhs[0]  += 0.5*dt*(K_mat[0,0]*1.0)   # 左边界 x=0 恒 1
    b_rhs[-1] = 0.0                        # 右边界 x=x_max 恒 0
    g_pay = payoff                         # 下界向量

    # SSN 迭代
    for k in range(20):
        r = A@u_ssn - b_rhs
        r[active] = (u_ssn - g_pay)[active]
        J = A.copy()
        J[active] = 0.0
        J[active, active] = 1.0
        du = spsolve(J, -r)
        u_ssn += du
        new_active = (u_ssn - g_pay <= 0.0)
        if np.array_equal(new_active, active) and np.linalg.norm(du, np.inf) < 1e-6:
            break
        active = new_active

    # 边界保险
    u_ssn[0], u_ssn[-1] = 1.0, 0.0

# ========= 计时结束 +===========
end_total = time.time()
total_time = end_total - start_total
print("\n===== 计算时间统计 =====")
print(f"总耗时: {total_time:.4f} 秒")    

# ========== 结果比较 ==========
plt.figure(figsize=(10, 6))
plt.plot(x_nodal*K, v_cn*K, label='hp-FEM CN p='+str(p))
plt.plot(x_nodal*K, u_ssn*K, label='hp-FEM SSN p='+str(p), linestyle='--')
plt.axvline(K, ls='--', color='gray', label='K='+str(K))
plt.xlabel('S'); plt.ylabel('V')
plt.legend(); plt.title('CN方法与SSN方法结果比较')
plt.grid(True, alpha=0.3)
plt.show()
