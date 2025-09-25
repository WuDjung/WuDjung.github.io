"""
经典美式看跌 Crank-Nicolson + PSOR  (实践网格 M=400 N=2000)
"""
import numpy as np
from scipy.sparse import diags
from scipy.stats import norm

# ==================== 参数 ====================
K = 100.0          # 执行价
T = 1.0            # 到期年
r = 0.05           # 无风险
sigma = 0.2        # 波动率
x_max = 3.0        # S 上限 3K
M = 400            # 空间格
N = 2000           # 时间步
omega = 1.5        # PSOR 松弛
MAX_ITER = 50
TOL = 1e-6
# ============================================

dx = x_max / M
dt = T / N
x = np.linspace(0.0, x_max, M+1)

#  payoff 向量（无量纲）
payoff = np.maximum(1.0 - x, 0.0)

# 构建矩阵 L 与 CN 左右矩阵 A, B
alpha = 0.5 * sigma**2 * x**2
beta  = r * x
a = alpha[1:-1]/dx**2 - beta[1:-1]/(2*dx)
b = -2*alpha[1:-1]/dx**2 - r
c = alpha[1:-1]/dx**2 + beta[1:-1]/(2*dx)

L = diags([a, b, c], [-1, 0, 1], shape=(M-1, M-1))
I = diags([np.ones(M-1)], [0])
A = I - 0.5*dt*L
B = I + 0.5*dt*L
A = A.tocsr()
B = B.tocsr()

# 边界贡献向量（右端用）
b_left  = 0.5*dt*a[0]  * payoff[0]
b_right = 0.5*dt*c[-1] * payoff[-1]

# ==================== 主循环 ====================
v = payoff.copy()
for n in range(N):
    # 右端 rhs = B*v + 边界
    rhs = B.dot(v[1:-1])
    rhs[0]  += b_left
    rhs[-1] += b_right

    # PSOR 迭代
    v_old = v.copy()
    for it in range(MAX_ITER):
        # 调整循环范围，避免超出A矩阵的维度
        for j in range(1, M):
            jm, jp = j-1, j+1
            # 计算临时值，修正边界检查条件
            term1 = A[jm, jm-1] * v[jm] if jm > 0 else 0.0
            term2 = A[jm, jp-1] * v[jp] if (jp-1 < M-1) else 0.0  # 修复索引检查
            temp = (rhs[jm] - term1 - term2) / A[jm, jm]
            v[j] = max(payoff[j], v_old[j] + omega*(temp - v_old[j]))
        if np.linalg.norm(v - v_old, ord=np.inf) < TOL:
            break
        v_old = v.copy()
    # 边界固定
    v[0], v[-1] = payoff[0], payoff[-1]

# ==================== 输出 ====================
print('实践网格V1(M=400,N=2000)  在 S=K:')
idx_K = np.argmin(np.abs(x - 1.0))
print('V(S=K)/K = {:.6f}'.format(v[idx_K]))
