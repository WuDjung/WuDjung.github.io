# hp_fem_adaptive.py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from skfem import MeshLine  # 用于网格操作
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
M = 10               # 初始单元数（自适应会加密）
N = 500              # 时间步
p_init = 1           # 初始单元阶数
p_max = 4            # 最大阶数限制
MAX_ADAPT = 5        # 最大自适应步数
TOL = 1e-3           # 误差容限
# ==========================

dt = T / N
# 初始化网格（使用scikit-fem的MeshLine）
mesh = MeshLine(np.linspace(0, x_max, M+1))
p_list = np.array([p_init for _ in range(mesh.nelements)])  # 各单元阶数列表

# ========== 局部基函数（一维 hierarchical） ==========
def local_matrices(xl, xr, p):
    h = xr - xl
    # 高斯点 & 权重（p+1 点即可精确 2p 阶）
    gp, gw = np.polynomial.legendre.leggauss(p+1)
    gp = 0.5*(gp + 1)*h + xl      # 映射到单元 [xl, xr]
    gw *= 0.5*h                   # 权重缩放

    n_loc = p + 1                 # 局部自由度
    Ke = np.zeros((n_loc, n_loc))
    Me = np.zeros_like(Ke)

    for xi, wi in zip(gp, gw):
        # 基函数及其导数（hierarchical模式）
        phi = np.array([1.0, (xi - xl)/(xr - xl) - 0.5])  # 线性项
        if p >= 2:
            L2 = 0.5*(3*((xi - xl)/(xr - xl))**2 - 1)
            phi = np.append(phi, L2)
        if p >= 3:
            L3 = 0.5*(5*((xi - xl)/(xr - xl))**3 - 3*((xi - xl)/(xr - xl)))
            phi = np.append(phi, L3)
        
        # 导数（对x的导数）
        dphi = np.array([0.0, 1/(xr - xl)])  # 线性项导数
        if p >= 2:
            dphi = np.append(dphi, 3*((xi - xl)/(xr - xl))/(xr - xl))
        if p >= 3:
            dphi = np.append(dphi, 0.5*(15*((xi - xl)/(xr - xl))**2 - 3)/(xr - xl))

        # 局部矩阵装配
        Ke += 0.5*sigma**2*xi**2 * np.outer(dphi, dphi) * wi
        Ke += r*xi * np.outer(dphi, phi) * wi
        Ke -= r * np.outer(phi, phi) * wi
        Me += np.outer(phi, phi) * wi
    return Ke, Me

# ========== 全局装配（适应变网格和变阶数） ==========
def assemble_hp_matrices(mesh, p_list):
    dof_map = []               # 单元→全局自由度映射
    tot_dof = 0
    # 遍历所有单元
    for e in range(mesh.nelements):
        p = p_list[e]
        dof_map.append(np.arange(tot_dof, tot_dof + p + 1))
        tot_dof += p + 1
    
    K_mat = csr_matrix((tot_dof, tot_dof))
    M_mat = csr_matrix((tot_dof, tot_dof))
    
    # 单元循环装配
    for e in range(mesh.nelements):
        xl = mesh.p[0, e]
        xr = mesh.p[0, e+1]
        p = p_list[e]
        Ke, Me = local_matrices(xl, xr, p)
        loc = dof_map[e]
        K_mat[np.ix_(loc, loc)] += Ke
        M_mat[np.ix_(loc, loc)] += Me
    
    return K_mat, M_mat, dof_map, tot_dof

# ========== 求解LCP（SSN方法） ==========
def solve_lcp_ssn(K_mat, M_mat, dof_map, tot_dof, x_nodal, v0, dt):
    A = M_mat - 0.5*dt*K_mat
    B = M_mat + 0.5*dt*K_mat
    payoff = np.maximum(1.0 - x_nodal, 0.0)
    
    u = v0.copy()
    active = np.zeros_like(u, dtype=bool)
    
    for n in range(N):
        b_rhs = B @ u
        b_rhs[0] += 0.5*dt*(K_mat[0,0]*1.0)  # 左边界处理
        b_rhs[-1] = 0.0                       # 右边界处理
        
        # SSN迭代
        for k in range(20):
            r = A @ u - b_rhs
            r[active] = (u - payoff)[active]
            J = A.copy()
            J[active] = 0.0
            J[active, active] = 1.0
            du = spsolve(J, -r)
            u += du
            new_active = (u - payoff <= 1e-8)  # 松弛判断
            
            if np.array_equal(new_active, active) and np.linalg.norm(du, np.inf) < 1e-6:
                break
            active = new_active
        
        u[0], u[-1] = 1.0, 0.0  # 强制边界
    
    return u

# ========== 误差指标计算 ==========
def compute_indicator(u, mesh, p_list, K_mat, M_mat, dt):
    # 简化的残差型误差指标
    n_elem = mesh.nelements
    eta_e = np.zeros(n_elem)
    for e in range(n_elem):
        xl = mesh.p[0, e]
        xr = mesh.p[0, e+1]
        h_e = xr - xl
        # 粗略估计单元误差（可根据需要细化）
        eta_e[e] = h_e * np.linalg.norm(u[e*2:(e+1)*2])  # 简化计算
    return eta_e / np.max(eta_e)  # 归一化

# ========== Dorfer标记策略 ==========
def dorfer_mark(eta_e, theta=0.6):
    # 按误差排序，标记累积误差超过theta的单元
    sorted_idx = np.argsort(eta_e)[::-1]  # 降序排列
    total_error = np.sum(eta_e)
    cumulative = 0.0
    mark = []
    for idx in sorted_idx:
        cumulative += eta_e[idx]
        mark.append(idx)
        if cumulative / total_error >= theta:
            break
    return mark

# ========== hp-refine实现 ==========
def hp_refine(mesh, p_list, mark):
    new_p = p_list.copy()
    # 1. 先对可升阶的单元执行“升阶”
    for e in mark:
        if new_p[e] < p_max:
            new_p[e] += 1  # 单元e的阶数+1
        else:
            # 阶数已达上限，标记为“需要加密”
            pass
    
    # 2. 处理“需要加密”的单元（阶数无法再升）
    refine_elems = [e for e in mark if new_p[e] >= p_max]
    if refine_elems:
        old_nelems = mesh.nelements  # 加密前的单元总数
        mesh = mesh.refined(refine_elems)  # 执行网格加密
        new_nelems = mesh.nelements  # 加密后的单元总数
        
        # 初始化新的阶数列表（长度为新单元数）
        new_p_list = np.zeros(new_nelems, dtype=int)
        
        # ① 处理“未被加密”的单元：阶数保持不变
        for e in range(old_nelems):
            if e not in refine_elems:
                new_p_list[e] = new_p[e]
        
        # ② 处理“被加密”的单元：新生成的单元继承父单元阶数
        for e in refine_elems:
            # 原单元e加密后，会分裂为两个新单元（索引为e和e+1）
            new_p_list[e] = new_p[e]
            new_p_list[e+1] = new_p[e]
        
        new_p = new_p_list  # 更新为新的阶数列表
    
    return mesh, new_p
# ========== 自适应循环 ==========
adapt_results = []
for adapt_step in range(MAX_ADAPT):
    print(f"自适应步骤 {adapt_step+1}/{MAX_ADAPT}")
    
    # 装配矩阵
    K_mat, M_mat, dof_map, tot_dof = assemble_hp_matrices(mesh, p_list)
    
    # 生成节点坐标（用于初始条件和绘图）
    x_nodal = np.linspace(0, x_max, tot_dof)
    v0 = np.maximum(1.0 - x_nodal, 0.0)
    
    # 求解LCP
    u = solve_lcp_ssn(K_mat, M_mat, dof_map, tot_dof, x_nodal, v0, dt)
    adapt_results.append((x_nodal, u, mesh.nelements, p_list.copy()))
    
    # 计算误差指标
    eta_e = compute_indicator(u, mesh, p_list, K_mat, M_mat, dt)
    print(f"最大单元误差: {np.max(eta_e):.6f}")
    
    # 检查收敛
    if np.max(eta_e) < TOL:
        print(f"在步骤 {adapt_step+1} 达到收敛条件")
        break
    
    # 标记和细化
    mark = dorfer_mark(eta_e)
    mesh, p_list = hp_refine(mesh, p_list, mark)

# ========= 计时结束 +===========
end_total = time.time()
total_time = end_total - start_total
print("\n===== 计算时间统计 =====")
print(f"总耗时: {total_time:.4f} 秒")    

# ========== 结果可视化 ==========
plt.figure(figsize=(12, 8))

# 绘制最终结果
final_x, final_u, _, _ = adapt_results[-1]
plt.plot(final_x*K, final_u*K, 'b-', label=f'最终结果 (自适应步骤 {len(adapt_results)})')

# 绘制初始结果（用于对比）
init_x, init_u, _, _ = adapt_results[0]
plt.plot(init_x*K, init_u*K, 'k--', alpha=0.5, label=f'初始结果 (单元数 {adapt_results[0][2]})')

plt.axvline(K, ls='--', color='gray', label='执行价 K')
plt.xlabel('标的资产价格 S')
plt.ylabel('期权价值 V')
plt.title('自适应hp-FEM求解结果')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 打印自适应过程信息
print("\n自适应过程总结:")
for i, (_, _, nelems, p) in enumerate(adapt_results):
    avg_p = np.mean(p)
    print(f"步骤 {i+1}: 单元数={nelems}, 平均阶数={avg_p:.2f}")

