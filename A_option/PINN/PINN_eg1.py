# 1. 导入依赖库
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc  # 拉丁超立方采样（论文指定）
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 2. 全局配置（复刻论文参数）
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)  # 固定随机种子，保证复现性
np.random.seed(42)

# 3. 生成训练数据（论文设置：N_u=100个数据点，N_f=10000个配点）
def generate_burgers_data(N_u=100, N_f=10000):
    """
    生成两类数据：
    - 数据损失点（N_u）：初始条件+边界条件的标注数据
    - 物理损失点（N_f）：PDE定义域内的随机配点（用于计算残差）
    """
    # 3.1 数据损失点（MSE_u）：初始条件+边界条件
    # 边界条件：x=-1 和 x=1，t∈[0,1]，u=0
    x_bc = np.concatenate([np.full((N_u//2, 1), -1.0), np.full((N_u//2, 1), 1.0)], axis=0)
    t_bc = np.random.rand(N_u, 1)  # t随机采样
    u_bc = np.zeros_like(x_bc)     # 边界值u=0

    # 初始条件：t=0，x∈[-1,1]，u=-sin(πx)
    x_ic = np.random.rand(N_u//2, 1) * 2 - 1  # x∈[-1,1]
    t_ic = np.zeros_like(x_ic)                 # t=0
    u_ic = -np.sin(np.pi * x_ic)               # 初始值

    # 合并数据损失点（x, t, u）
    x_u = np.concatenate([x_bc, x_ic], axis=0)
    t_u = np.concatenate([t_bc, t_ic], axis=0)
    u_u = np.concatenate([u_bc, u_ic], axis=0)

    # 3.2 物理损失点（MSE_f）：拉丁超立方采样（论文指定，空间填充性更好）
    sampler = qmc.LatinHypercube(d=2)  # 2维采样（x, t）
    sample = sampler.random(n=N_f)     # 采样N_f个点，范围[0,1]
    x_f = 2 * sample[:, 0:1] - 1       # x映射到[-1,1]
    t_f = sample[:, 1:2]               # t保持[0,1]

    # 转换为PyTorch张量并移至设备（CPU/GPU）
    x_u_tensor = torch.tensor(x_u, dtype=torch.float32).to(device)
    t_u_tensor = torch.tensor(t_u, dtype=torch.float32).to(device)
    u_u_tensor = torch.tensor(u_u, dtype=torch.float32).to(device)
    x_f_tensor = torch.tensor(x_f, dtype=torch.float32).to(device)
    t_f_tensor = torch.tensor(t_f, dtype=torch.float32).to(device)

    return (x_u_tensor, t_u_tensor, u_u_tensor), (x_f_tensor, t_f_tensor)

# 4. 定义PINN网络（复刻论文：9层MLP，每层20神经元，Tanh激活）
class BurgersPINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=20, hidden_layers=9, output_dim=1):
        super().__init__()
        layers = []
        # 输入层：2维（x, t）→ hidden_dim维
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())  # 论文用Tanh激活，拟合非线性解更稳定
        # 隐藏层：9层（论文指定）
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        # 输出层：hidden_dim维 → 1维（u(t,x)）
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x, t):
        """前向传播：输入x和t，输出u(t,x)"""
        input_tensor = torch.cat([x, t], dim=1)  # 拼接x和t（batch_size×2）
        return self.model(input_tensor)

# 5. 定义损失函数（论文双损失：MSE_u + MSE_f）
def compute_loss(model, x_u, t_u, u_u, x_f, t_f):
    """
    计算总损失：
    - MSE_u：数据损失（拟合初始/边界数据）
    - MSE_f：物理损失（惩罚PDE残差，强制满足物理定律）
    """
    # 5.1 数据损失 MSE_u
    u_pred = model(x_u, t_u)
    mse_u = torch.mean((u_pred - u_u) ** 2)

    # 5.2 物理损失 MSE_f：自动微分计算PDE残差
    # 开启自动微分（需计算x_f和t_f的导数）
    x_f.requires_grad = True
    t_f.requires_grad = True
    u_f = model(x_f, t_f)  # 网络预测配点处的u值

    # 自动微分计算一阶导数 u_t（对t的偏导）、u_x（对x的偏导）
    u_t = torch.autograd.grad(
        outputs=u_f,
        inputs=t_f,
        grad_outputs=torch.ones_like(u_f),
        create_graph=True,  # 保留计算图，用于高阶导数
        retain_graph=True
    )[0]

    u_x = torch.autograd.grad(
        outputs=u_f,
        inputs=x_f,
        grad_outputs=torch.ones_like(u_f),
        create_graph=True,
        retain_graph=True
    )[0]

    # 自动微分计算二阶导数 u_xx（对x的二阶偏导）
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x_f,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0]

    # 计算PDE残差 f = u_t + u*u_x - (0.01/π)*u_xx（论文公式3）
    nu = 0.01 / np.pi  # 粘度参数
    f = u_t + u_f * u_x - nu * u_xx
    mse_f = torch.mean(f ** 2)

    # 总损失（论文未加权，直接求和）
    total_loss = mse_u + mse_f
    return total_loss, mse_u, mse_f

# 6. 训练PINN模型（复刻论文：L-BFGS优化器）
def train_pinn(model, x_u, t_u, u_u, x_f, t_f, epochs=1000):
    """训练PINN，用L-BFGS优化（论文指定，小数据场景收敛更快）"""
    # 定义优化器（L-BFGS比Adam更适合小数据、凸优化场景）
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1e-3,
        max_iter=50000,  # 最大迭代次数
        max_eval=50000,
        history_size=50,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe"
    )

    # 训练循环（L-BFGS需自定义闭包函数）
    def closure():
        optimizer.zero_grad()  # 清零梯度
        total_loss, mse_u, mse_f = compute_loss(model, x_u, t_u, u_u, x_f, t_f)
        total_loss.backward()  # 反向传播
        return total_loss

    # 开始训练
    print("开始训练PINN模型...")
    for epoch in range(epochs):
        optimizer.step(closure)  # L-BFGS一步迭代=多次函数评估

        # 每100次迭代打印日志
        if (epoch + 1) % 100 == 0:
            total_loss, mse_u, mse_f = compute_loss(model, x_u, t_u, u_u, x_f, t_f)
            print(f"Epoch [{epoch+1}/{epochs}] | 总损失: {total_loss.item():.8f} | 数据损失: {mse_u.item():.8f} | 物理损失: {mse_f.item():.8f}")

    print("训练完成！")
    return model

# 7. 可视化结果（复刻论文图1：时空热力图+时间快照对比）
def visualize_results(model):
    """
    可视化3部分内容：
    1. 3D时空曲面图（u-t-x）
    2. 2D热力图（复刻论文图1上半部分）
    3. 时间快照对比（t=0.25,0.5,0.75，复刻论文图1下半部分）
    """
    # 生成均匀网格（用于可视化全时空域）
    x = np.linspace(-1, 1, 200)  # x轴200个点
    t = np.linspace(0, 1, 200)   # t轴200个点
    X, T = np.meshgrid(x, t)      # 生成网格

    # 转换为张量并预测
    X_tensor = torch.tensor(X.reshape(-1, 1), dtype=torch.float32).to(device)
    T_tensor = torch.tensor(T.reshape(-1, 1), dtype=torch.float32).to(device)
    with torch.no_grad():  # 推理时禁用梯度计算
        U_pred = model(X_tensor, T_tensor).cpu().numpy().reshape(X.shape)

    # 计算解析解（用于对比，论文提供解析解公式）
    def burgers_analytic(x, t):
        nu = 0.01 / np.pi
        return -np.sin(np.pi * (x - 2 * nu * np.pi * t)) / np.cosh(np.pi * (x - 2 * nu * np.pi * t) / 2)

    U_analytic = burgers_analytic(X, T)

    # 计算相对L2误差（复刻论文评估指标）
    l2_error = np.linalg.norm(U_pred - U_analytic, 2) / np.linalg.norm(U_analytic, 2)
    print(f"\n相对L2误差: {l2_error:.6f}（论文结果：6.7e-4）")

    # 绘制3D时空曲面图
    fig = plt.figure(figsize=(18, 5))

    # 子图1：3D预测解
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, T, U_pred, cmap=cm.viridis, edgecolor='none')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(t,x)')
    ax1.set_title('PINN预测解（3D）')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # 子图2：2D热力图（复刻论文图1上半部分）
    ax2 = fig.add_subplot(1, 3, 2)
    contour = ax2.contourf(T, X, U_pred, levels=50, cmap=cm.viridis)
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_title('PINN预测解（热力图）')
    fig.colorbar(contour, ax=ax2)

    # 子图3：时间快照对比（t=0.25,0.5,0.75）
    ax3 = fig.add_subplot(1, 3, 3)
    t_snapshots = [0.25, 0.5, 0.75]
    colors = ['r-', 'g-', 'b-']
    labels = [f't={t}' for t in t_snapshots]
    for i, t_val in enumerate(t_snapshots):
        t_idx = np.argmin(np.abs(t - t_val))  # 找到对应时间索引
        ax3.plot(x, U_pred[t_idx, :], colors[i], label=labels[i], linewidth=2)
        ax3.plot(x, U_analytic[t_idx, :], colors[i], linestyle='--', alpha=0.8, linewidth=2)
    ax3.set_xlabel('x')
    ax3.set_ylabel('u(t,x)')
    ax3.set_title('时间快照对比（实线=预测，虚线=解析解）')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 8. 主函数（串联全流程）
if __name__ == "__main__":
    # 生成数据（论文参数：N_u=100，N_f=10000）
    (x_u, t_u, u_u), (x_f, t_f) = generate_burgers_data(N_u=100, N_f=10000)
    
    # 初始化模型（复刻论文：9层MLP，每层20神经元）
    model = BurgersPINN(hidden_dim=20, hidden_layers=9)
    
    # 训练模型
    trained_model = train_pinn(model, x_u, t_u, u_u, x_f, t_f, epochs=1000)
    
    # 可视化结果
    visualize_results(trained_model)