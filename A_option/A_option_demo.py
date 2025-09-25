# american_ipinn_demo.py
import torch, numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 参数
r, sigma, K, T = 0.05, 0.2, 100, 1.0
N_SAMPLE = 20000
LAYER  = [2] + 3*[64] + [1]   # 网络结构

# 生成训练点
def sample(N):
    xi = torch.rand(N, 1, device=device)          # xi in [0,1]
    tau  = torch.rand(N, 1, device=device)        # tau in [0,1]
    return xi, tau

# front-fixing 变换
def xi_to_S(xi, tau):
    """ S = B(tau) * xi,  B(tau) 是待求自由边界 """
    B = B_net(tau)          # 网络输出 B(tau)
    return B * xi, B

#  payoff
def psi(S):
    return torch.relu(K - S)

# 神经网络
class MLP(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        net = []
        for i in range(len(layers)-2):
            net += [torch.nn.Linear(layers[i], layers[i+1]), torch.nn.Tanh()]
        net += [torch.nn.Linear(layers[-2], layers[-1])]
        self.net = torch.nn.Sequential(*net)
    def forward(self, x):
        return self.net(x)

V_net  = MLP(LAYER).to(device)   # 逼近 V(xi,tau)
B_net  = torch.nn.Sequential(
            torch.nn.Linear(1,32), torch.nn.Tanh(),
            torch.nn.Linear(32,1), torch.nn.Sigmoid()
         ).to(device)             # 输出 B(tau)/K 归一化
for p in B_net.parameters():
    torch.nn.init.constant_(p, 0)
B_net[-2].bias.data.fill_(0.9)    # 初始 B≈K

# 自动微分工具
def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]

# PDE 残差 (变换后的 BS 方程)
def pde_residual(xi, tau):
    xi.requires_grad_(True)
    tau.requires_grad_(True)
    S, B = xi_to_S(xi, tau)
    V = V_net(torch.cat([xi, tau], 1))
    V_xi = grad(V, xi)
    V_tau = grad(V, tau)
    V_S = V_xi / B
    V_SS = grad(V_S, xi) / B**2
    # 变换后的 PDE:  V_tau = 0.5 sigma^2 xi^2 V_xixi + (r - 0.5 sigma^2) xi V_xi - r V
    # 但用 S 语言写更直观
    lhs = V_tau
    rhs = 0.5*sigma**2*S**2*V_SS + r*S*V_S - r*V
    return lhs - rhs

# Loss 组成
def loss():
    xi, tau = sample(N_SAMPLE)
    # 1. PDE 残差
    res = pde_residual(xi, tau)
    loss_pde = torch.mean(res**2)
    # 2. 边界条件
    tau0 = torch.zeros_like(xi)
    S0, _ = xi_to_S(xi, tau0)
    V0 = V_net(torch.cat([xi, tau0], 1))
    payoff0 = psi(S0)
    loss_terminal = torch.mean((V0 - payoff0)**2)
    # 3. 自由边界条件  V(xi=1, tau)=K-B(tau)  且  V_xi(xi=1,tau)=-1
    xi1 = torch.ones_like(tau, requires_grad=True)
    S1, B1 = xi_to_S(xi1, tau)
    V1 = V_net(torch.cat([xi1, tau], 1))
    V1_xi = grad(V1, xi1)
    loss_bc1 = torch.mean((V1 - (K - B1))**2)
    loss_bc2 = torch.mean((V1_xi + 1)**2)
    # 4. 互补条件  V >= payoff
    S_all, _ = xi_to_S(xi, tau)
    V_all = V_net(torch.cat([xi, tau], 1))
    payoff_all = psi(S_all)
    comp = torch.relu(payoff_all - V_all)
    loss_comp = torch.mean(comp**2)
    return loss_pde + loss_terminal + loss_bc1 + loss_bc2 + loss_comp

# 训练
optimizer = torch.optim.Adam(list(V_net.parameters())+list(B_net.parameters()), lr=5e-3)
for epoch in range(5000):
    optimizer.zero_grad()
    l = loss()
    l.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(epoch, l.item())

with torch.no_grad():
    tau  = torch.linspace(0, 1, 100).view(-1,1).to(device)
    xi   = torch.linspace(0.01,1, 100).view(-1,1).to(device)
    Tau, Xi = torch.meshgrid(tau.squeeze(), xi.squeeze(), indexing='ij')  # torch≥1.10
    Xi_flat  = Xi.reshape(-1,1)
    Tau_flat = Tau.reshape(-1,1)
    V_pred = V_net(torch.cat([Xi_flat, Tau_flat],1)).reshape(100,100).cpu()
    B_pred = B_net(tau).cpu()*K                 # shape (100,1)
    S_grid = (B_pred @ xi.T)                    # shape (100,100)

    # 取 t=0（tau=1）截面
    S_line = S_grid[-1].squeeze()               # (100,)
    V_line = V_pred[-1]                         # (100,)
    payoff_line = np.maximum(K - S_line.numpy(), 0)

    plt.figure(figsize=(6,4))
    plt.plot(S_line, V_line,   label='IPINN t=0')
    plt.plot(S_line, payoff_line, '--', label='payoff')
    plt.xlabel('S'); plt.ylabel('V'); plt.legend()
    plt.title('American Put, IPINN+front-fixing')
    plt.show()