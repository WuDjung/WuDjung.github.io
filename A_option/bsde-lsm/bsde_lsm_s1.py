import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc, norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# 设置随机种子，保证结果可复现
np.random.seed(1234)
torch.manual_seed(1234)

# 期权参数
K = 100.0  # 行权价
T = 1.0    # 到期时间（年）
r = 0.05   # 无风险利率
sigma = 0.2  # 波动率
V_ext = 8.0229  # hp-FEM外推基准值

def generate_paths(N_path, N_step):
    """生成股票价格路径"""
    dt = T / N_step
    
    # 使用Sobol低差异序列
    sobol = qmc.Sobol(d=N_step, scramble=True, seed=1234)
    # 确保N_path是2的幂次
    m = int(np.ceil(np.log2(N_path)))
    Z = sobol.random_base2(m=m)[:N_path]
    dW = norm.ppf(Z) * np.sqrt(dt)
    
    # 生成路径
    S = np.zeros((N_path, N_step+1))
    S[:, 0] = K  # 初始价格设为行权价
    
    for n in range(N_step):
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * dW[:, n]
        S[:, n+1] = S[:, n] * np.exp(drift + diffusion)
    
    return S, dt

def lsm_pricing(S, dt, N_step):
    """使用LSM方法计算期权价格"""
    N_path = S.shape[0]
    
    # 倒推计算现金流
    cashflow = np.maximum(K - S[:, -1], 0)  # 到期时的现金流
    
    # 存储每个时间步的continuation value用于后续训练
    C_lsm = np.zeros_like(S)
    C_lsm[:, -1] = 0.0  # 到期时没有继续持有价值
    
    # 存储最优行权边界的估计
    exercise_boundary = np.full(N_step+1, np.nan)
    
    for n in range(N_step-1, -1, -1):
        # 仅对价内路径进行回归
        ITM = (S[:, n] < K)
        X = S[ITM, n].reshape(-1, 1)
        y = cashflow[ITM] * np.exp(-r * dt)
        
        # 使用三次多项式基函数
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        
        # 线性回归
        reg = LinearRegression().fit(X_poly, y)
        C = reg.predict(X_poly)  # continuation value
        
        # 记录LSM的continuation value
        C_lsm[ITM, n] = C
        C_lsm[~ITM, n] = 0.0  # 价外期权不会行权，继续持有价值为0
        
        # 确定最优行权决策
        exercise_value = K - S[ITM, n]
        exercise = exercise_value > C
        
        # 更新现金流
        cashflow[ITM] = np.where(exercise, exercise_value, cashflow[ITM])
        cashflow[ITM] *= np.exp(-r * dt)  # 折现到上一时间步
        
        # 估计最优行权边界（在ITM区域中找到行权与不行权的临界点）
        if np.any(ITM):
            sorted_idx = np.argsort(S[ITM, n])
            sorted_S = S[ITM, n][sorted_idx]
            sorted_exercise = exercise[sorted_idx]
            
            # 找到从不行权到行权的转变点
            transition = np.where(np.diff(sorted_exercise.astype(int)) == 1)[0]
            if len(transition) > 0:
                exercise_boundary[n] = (sorted_S[transition[-1]] + sorted_S[transition[-1]+1]) / 2
            elif np.all(sorted_exercise):
                exercise_boundary[n] = np.min(sorted_S)
            else:
                exercise_boundary[n] = np.max(sorted_S)
    
    # 计算期权价格
    V_lsm = np.mean(cashflow * np.exp(-r * dt))
    
    return V_lsm, C_lsm, exercise_boundary

class ContinuationValueNet(nn.Module):
    """用于预测continuation value的神经网络"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 输入：t和S
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)   # 输出：continuation value
        )
    
    def forward(self, t, S):
        # 确保输入形状正确
        t = t.view(-1, 1)
        S = S.view(-1, 1)
        x = torch.cat([t, S], dim=1)
        return self.net(x).squeeze()

def deep_bsde_training(S, C_lsm, dt, N_step, epochs=500, lr=1e-3):
    """训练Deep BSDE模型"""
    N_path = S.shape[0]
    
    # 转换为PyTorch张量
    t_grid = torch.linspace(0, T, N_step+1, dtype=torch.float32)
    S_tensor = torch.tensor(S, dtype=torch.float32, requires_grad=True)
    C_target = torch.tensor(C_lsm, dtype=torch.float32)
    
    # 初始化网络和优化器
    net = ContinuationValueNet()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    
    # 训练循环
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0.0
        
        # 计算每个时间步的损失
        for n in range(N_step):
            t_n = t_grid[n].repeat(N_path)
            C_pred = net(t_n, S_tensor[:, n])
            loss += torch.mean((C_pred - C_target[:, n])** 2)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
    
    # 计算期权价格
    with torch.no_grad():
        t0 = torch.tensor(0.0, dtype=torch.float32).repeat(N_path)
        C0_pred = net(t0, S_tensor[:, 0])
        exercise_value0 = K - S_tensor[:, 0]
        cashflow0 = torch.max(exercise_value0, C0_pred)
        V_dbsd = torch.mean(cashflow0 * torch.exp(-r * dt)).item()
    
    # 计算Delta (∂C/∂S)
    with torch.enable_grad():
        t0 = torch.tensor(0.0, dtype=torch.float32)
        S0 = torch.tensor(K, dtype=torch.float32, requires_grad=True)
        C0 = net(t0, S0)
        delta = torch.autograd.grad(C0, S0, create_graph=True)[0].item()
        # 美式期权的Delta是行权价值和继续持有价值中较大者的导数
        exercise_value = K - S0
        if exercise_value > C0:
            delta = -1.0  # 行权时Delta为-1
    
    # 计算最优行权边界
    exercise_boundary_dbsd = np.full(N_step+1, np.nan)
    with torch.no_grad():
        for n in range(N_step+1):
            t_n = torch.tensor(t_grid[n], dtype=torch.float32)
            
            # 在可能的边界附近搜索
            S_candidates = torch.linspace(50, K, 1000, dtype=torch.float32)
            C_pred = net(t_n, S_candidates)
            exercise_value = K - S_candidates
            
            # 找到行权和不行权的临界点
            diff = exercise_value - C_pred
            crossing = torch.where(torch.diff(torch.sign(diff)))[0]
            
            if len(crossing) > 0:
                # 线性插值找到精确的边界点
                idx = crossing[-1]
                S1, S2 = S_candidates[idx], S_candidates[idx+1]
                d1, d2 = diff[idx], diff[idx+1]
                exercise_boundary_dbsd[n] = S1 - (d1 * (S2 - S1)) / (d2 - d1)
    
    return V_dbsd, delta, exercise_boundary_dbsd, net, losses

def convergence_analysis():
    """分析不同路径数下的收敛性能"""
    path_nums = [10000, 50000, 100000, 500000, 1000000]
    N_step = 100
    err_lsm = []
    err_dbsd = []
    delta_lsm_list = []
    delta_dbsd_list = []
    
    print("进行收敛性分析...")
    for N in path_nums:
        print(f"路径数: {N}")
        
        # 生成路径
        S, dt = generate_paths(int(N), N_step)
        
        # LSM定价
        V_lsm, C_lsm, _ = lsm_pricing(S, dt, N_step)
        
        # 计算LSM的Delta（使用中心差分）
        epsilon = 0.01
        S_plus, _ = generate_paths(int(N), N_step)
        S_plus[:, 0] = K + epsilon
        V_lsm_plus, _, _ = lsm_pricing(S_plus, dt, N_step)
        
        S_minus, _ = generate_paths(int(N), N_step)
        S_minus[:, 0] = K - epsilon
        V_lsm_minus, _, _ = lsm_pricing(S_minus, dt, N_step)
        
        delta_lsm = (V_lsm_plus - V_lsm_minus) / (2 * epsilon)
        
        # Deep BSDE定价
        V_dbsd, delta_dbsd, _, _, _ = deep_bsde_training(S, C_lsm, dt, N_step, epochs=300)
        
        # 记录误差
        err_lsm.append(abs(V_lsm - V_ext))
        err_dbsd.append(abs(V_dbsd - V_ext))
        delta_lsm_list.append(delta_lsm)
        delta_dbsd_list.append(delta_dbsd)
        
        print(f"LSM价格: {V_lsm:.5f}, 误差: {err_lsm[-1]:.6f}")
        print(f"Deep BSDE价格: {V_dbsd:.5f}, 误差: {err_dbsd[-1]:.6f}")
        print(f"LSM Delta: {delta_lsm:.4f}, Deep BSDE Delta: {delta_dbsd:.4f}")
        print("-" * 50)
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.loglog(path_nums, err_lsm, 'o-', label='LSM')
    plt.loglog(path_nums, err_dbsd, 's-', label='Deep BSDE-LSM')
    plt.loglog(path_nums, 1/np.sqrt(path_nums), ':', label='1/√N 参考线')
    
    plt.xlabel('路径数量', fontsize=12)
    plt.ylabel('价格误差', fontsize=12)
    plt.title('不同路径数下的价格误差收敛曲线', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('convergence_curve.png', dpi=300)
    plt.show()
    
    # 绘制Delta收敛曲线
    plt.figure(figsize=(10, 6))
    plt.semilogx(path_nums, delta_lsm_list, 'o-', label='LSM Delta')
    plt.semilogx(path_nums, delta_dbsd_list, 's-', label='Deep BSDE Delta')
    plt.axhline(y=-0.4402, color='r', linestyle=':', label='hp-FEM Delta基准')
    
    plt.xlabel('路径数量', fontsize=12)
    plt.ylabel('Delta值', fontsize=12)
    plt.title('不同路径数下的Delta收敛曲线', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('delta_convergence.png', dpi=300)
    plt.show()
    
    return path_nums, err_lsm, err_dbsd

def main():
    # 主要参数
    N_path = 1000000 # 百万路径
    N_step = 100        # 时间步数
    
    # Step 1: 生成路径并使用LSM计算
    print("Step 1: 运行LSM方法...")
    S, dt = generate_paths(N_path, N_step)
    V_lsm, C_lsm, exercise_boundary_lsm = lsm_pricing(S, dt, N_step)
    print(f"LSM Sobol 1M-path V(S=K) = {V_lsm:.5f}")
    
    # Step 2: 运行Deep BSDE-LSM混合方法
    print("\nStep 2: 训练Deep BSDE模型...")
    V_dbsd, delta_dbsd, exercise_boundary_dbsd, net, losses = deep_bsde_training(S, C_lsm, dt, N_step)
    print(f"Deep BSDE-LSM V(S=K) = {V_dbsd:.5f}")
    print(f"Deep BSDE-LSM Delta(S=K) = {delta_dbsd:.4f}")
    
    # 计算LSM的Delta（使用中心差分）
    print("\n计算LSM的Delta...")
    epsilon = 0.01
    S_plus, _ = generate_paths(N_path, N_step)
    S_plus[:, 0] = K + epsilon
    V_lsm_plus, _, _ = lsm_pricing(S_plus, dt, N_step)
    
    S_minus, _ = generate_paths(N_path, N_step)
    S_minus[:, 0] = K - epsilon
    V_lsm_minus, _, _ = lsm_pricing(S_minus, dt, N_step)
    
    delta_lsm = (V_lsm_plus - V_lsm_minus) / (2 * epsilon)
    print(f"LSM Delta(S=K) = {delta_lsm:.4f}")
    
    # Step 3: 显示三交叉验证结果
    print("\nStep 3: 三交叉验证结果")
    print(f"{'方法':<15} {'V(S=K)':<10} {'Δ(S=K)':<10} {'S*(0)':<10}")
    print("-" * 45)
    print(f"{'hp-FEM 外推':<15} {8.0229:<10.4f} {-0.4402:<10.4f} {77.38:<10.2f}")
    print(f"{'LSM 1M Sobol':<15} {V_lsm:<10.4f} {delta_lsm:<10.4f} {exercise_boundary_lsm[0]:<10.2f}")
    print(f"{'Deep BSDE':<15} {V_dbsd:<10.4f} {delta_dbsd:<10.4f} {exercise_boundary_dbsd[0]:<10.2f}")
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.title('Deep BSDE模型训练损失', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300)
    plt.show()
    
    # 绘制最优行权边界
    t_grid = np.linspace(0, T, N_step+1)
    plt.figure(figsize=(10, 6))
    plt.plot(t_grid, exercise_boundary_lsm, 'o-', label='LSM 行权边界', markersize=4)
    plt.plot(t_grid, exercise_boundary_dbsd, 's-', label='Deep BSDE 行权边界', markersize=4)
    plt.axhline(y=77.38, color='r', linestyle=':', label='hp-FEM 行权边界基准')
    plt.xlabel('时间 t', fontsize=12)
    plt.ylabel('最优行权边界 S*(t)', fontsize=12)
    plt.title('美式看跌期权的最优行权边界', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exercise_boundary.png', dpi=300)
    plt.show()
    
    # Step 4: 收敛性分析
    print("\nStep 4: 进行收敛性分析...")
    path_nums, err_lsm, err_dbsd = convergence_analysis()

if __name__ == "__main__":
    main()
