# ================================================================
# scheme2_mlp_phasefield.py
# 方案2: MLP学φ + MLP学V + Phase Field能量泛函 (纯MLP基线)
# ================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json, time, os, warnings
warnings.filterwarnings('ignore')

# 复用方案1的FDM和配置
from scheme1_cnn_phasefield import (
    Config, generate_fdm_solution, get_fdm_on_grid, SirenLayer, ValueMLP
)

config = Config()
torch.manual_seed(config.seed)
np.random.seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)


# ================================================================
# φ的MLP (替代CNN)
# ================================================================
class PhiMLP(nn.Module):
    """MLP: (S,t) -> φ ∈ (0,1)"""
    def __init__(self, hidden=128, layers=4, omega_0=10.0):
        super().__init__()
        nets = [SirenLayer(2, hidden, omega_0, is_first=True)]
        for _ in range(layers - 1):
            nets.append(SirenLayer(hidden, hidden, omega_0))
        self.net = nn.Sequential(*nets)
        self.last = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.last.weight.uniform_(
                -np.sqrt(6 / hidden) / omega_0,
                 np.sqrt(6 / hidden) / omega_0)
            self.last.bias.zero_()

    def forward(self, S, t):
        x = torch.cat([S, t], dim=-1)
        h = self.net(x)
        return torch.sigmoid(self.last(h))


# ================================================================
# Phase Field损失 (autograd版: φ的梯度也用autograd)
# ================================================================
class PhaseFieldLossAutograd:
    def __init__(self, cfg):
        self.cfg = cfg
        self.K = cfg.K
        self.r = cfg.r
        self.sigma = cfg.sigma
        self.eps = cfg.eps

    def compute_bs_residual(self, V, S, t):
        V_S = torch.autograd.grad(V, S, torch.ones_like(V),
                                  create_graph=True, retain_graph=True)[0]
        V_SS = torch.autograd.grad(V_S, S, torch.ones_like(V_S),
                                   create_graph=True, retain_graph=True)[0]
        V_t = torch.autograd.grad(V, t, torch.ones_like(V),
                                  create_graph=True, retain_graph=True)[0]
        return V_t + 0.5 * self.sigma**2 * S**2 * V_SS + self.r * S * V_S - self.r * V

    def compute_phi_gradient(self, phi, S, t):
        phi_S = torch.autograd.grad(phi, S, torch.ones_like(phi),
                                    create_graph=True, retain_graph=True)[0]
        phi_t = torch.autograd.grad(phi, t, torch.ones_like(phi),
                                    create_graph=True, retain_graph=True)[0]
        return phi_S, phi_t

    def energy(self, V, phi, LV, S, t):
        payoff = torch.clamp(self.K - S, min=0)
        L_bs = torch.mean(phi * LV**2)
        L_ex = torch.mean((1 - phi)**2 * (V - payoff)**2) / self.eps

        phi_S, phi_t = self.compute_phi_gradient(phi, S, t)
        W_phi = phi**2 * (1 - phi)**2
        L_int = self.eps * torch.mean(phi_S**2 + phi_t**2) + torch.mean(W_phi) / self.eps

        return L_bs, L_ex, L_int


# ================================================================
# 训练器
# ================================================================
class Scheme2Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        self.mlp_V = ValueMLP(cfg.mlp_hidden, cfg.mlp_layers, cfg.omega_0).to(self.device)
        self.mlp_phi = PhiMLP(hidden=128, layers=4, omega_0=10.0).to(self.device)
        self.loss_fn = PhaseFieldLossAutograd(cfg)

        # 网格点 (配点)
        S_g = torch.linspace(0, cfg.S_max, cfg.grid_S, device=self.device)
        t_g = torch.linspace(0, cfg.T, cfg.grid_t, device=self.device)
        Sm, tm = torch.meshgrid(S_g, t_g, indexing='ij')
        self.S_flat = Sm.reshape(-1, 1)
        self.t_flat = tm.reshape(-1, 1)
        self.S_mesh = Sm
        self.t_mesh = tm

        # FDM
        _, _, V_fdm = get_fdm_on_grid(cfg)
        self.V_fdm = torch.from_numpy(V_fdm).float().to(self.device)

        # 边界
        self._prepare_bc()
        self.history = {'loss': [], 'energy': [], 'arb': [], 'phi_acc': [], 'time': []}

    def _prepare_bc(self):
        cfg = self.cfg
        dev = self.device
        N = 200
        S_ic = torch.linspace(0, cfg.S_max, N, device=dev).unsqueeze(1)
        t_ic = torch.full_like(S_ic, cfg.T)
        V_ic = torch.clamp(cfg.K - S_ic, min=0)

        S_b0 = torch.zeros(N, 1, device=dev)
        t_b0 = torch.linspace(0, cfg.T, N, device=dev).unsqueeze(1)
        V_b0 = torch.full_like(S_b0, cfg.K)

        S_bm = torch.full((N, 1), cfg.S_max, device=dev)
        t_bm = torch.linspace(0, cfg.T, N, device=dev).unsqueeze(1)
        V_bm = torch.zeros_like(S_bm)

        self.bc_S = torch.cat([S_ic, S_b0, S_bm])
        self.bc_t = torch.cat([t_ic, t_b0, t_bm])
        self.bc_V = torch.cat([V_ic, V_b0, V_bm])

    def _bc_loss(self):
        S = self.bc_S.requires_grad_(True)
        t = self.bc_t.requires_grad_(True)
        return F.mse_loss(self.mlp_V(S, t), self.bc_V)

    def train_stage1(self):
        print("=" * 70)
        print("Stage 1: MLP_V预热 (FDM监督)")
        print("=" * 70)

        optimizer = AdamW(self.mlp_V.parameters(), lr=self.cfg.lr_mlp * 2, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.stage1_epochs, eta_min=1e-6)

        for epoch in range(self.cfg.stage1_epochs):
            self.mlp_V.train()
            optimizer.zero_grad()

            V_pred = self.mlp_V(self.S_flat, self.t_flat).reshape(self.cfg.grid_S, self.cfg.grid_t)
            loss = F.mse_loss(V_pred, self.V_fdm) + self.cfg.lambda_bc * self._bc_loss()
            loss.backward()
            nn.utils.clip_grad_norm_(self.mlp_V.parameters(), self.cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 500 == 0:
                rel = torch.mean(torch.abs(V_pred - self.V_fdm) / (self.V_fdm + 1e-6)).item()
                print(f"  Epoch {epoch+1}/{self.cfg.stage1_epochs} | Loss={loss.item():.6f} | RelErr={rel:.4f}")

    def train_stage2(self):
        print("\n" + "=" * 70)
        print("Stage 2: MLP_V + MLP_φ 联合训练")
        print("=" * 70)

        optimizer = AdamW([
            {'params': self.mlp_V.parameters(), 'lr': self.cfg.lr_mlp},
            {'params': self.mlp_phi.parameters(), 'lr': self.cfg.lr_cnn},
        ], weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.stage2_epochs, eta_min=1e-6)

        best_energy = float('inf')
        t0 = time.time()

        for epoch in range(self.cfg.stage2_epochs):
            self.mlp_V.train()
            self.mlp_phi.train()
            optimizer.zero_grad()

            S = self.S_flat.requires_grad_(True)
            t = self.t_flat.requires_grad_(True)

            V = self.mlp_V(S, t)
            phi = self.mlp_phi(S, t)
            LV = self.loss_fn.compute_bs_residual(V, S, t)

            L_bs, L_ex, L_int = self.loss_fn.energy(V, phi, LV, S, t)
            energy = L_bs + L_ex + L_int
            L_bc = self._bc_loss()

            if epoch < 3000:
                w_e = 1.0
            elif epoch < 8000:
                w_e = 5.0
            else:
                w_e = 10.0

            loss = w_e * energy + self.cfg.lambda_bc * L_bc
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.mlp_V.parameters()) + list(self.mlp_phi.parameters()),
                self.cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                V_g = V.reshape(self.cfg.grid_S, self.cfg.grid_t)
                phi_g = phi.reshape(self.cfg.grid_S, self.cfg.grid_t)
                payoff = torch.clamp(self.cfg.K - self.S_mesh, min=0)
                arb = torch.max(torch.clamp(payoff - V_g, min=0)).item()
                phi_target = (V_g > payoff + 0.01).float()
                phi_acc = ((phi_g > 0.5).float() == phi_target).float().mean().item()

            self.history['loss'].append(loss.item())
            self.history['energy'].append(energy.item())
            self.history['arb'].append(arb)
            self.history['phi_acc'].append(phi_acc)
            self.history['time'].append(time.time() - t0)

            if energy.item() < best_energy:
                best_energy = energy.item()
                torch.save({
                    'mlp_V': self.mlp_V.state_dict(),
                    'mlp_phi': self.mlp_phi.state_dict(),
                }, 'scheme2_best.pth')

            if (epoch + 1) % 1000 == 0:
                print(f"  Epoch {epoch+1}/{self.cfg.stage2_epochs} | "
                      f"E={energy.item():.4f} | Arb={arb:.4f} | φAcc={phi_acc:.3f}")

        ckpt = torch.load('scheme2_best.pth', weights_only=False)
        self.mlp_V.load_state_dict(ckpt['mlp_V'])
        self.mlp_phi.load_state_dict(ckpt['mlp_phi'])

    def train_stage3(self):
        print("\n" + "=" * 70)
        print("Stage 3: L-BFGS精调MLP_V (φ固定)")
        print("=" * 70)

        self.mlp_phi.eval()
        for p in self.mlp_phi.parameters():
            p.requires_grad = False

        optimizer = torch.optim.LBFGS(
            self.mlp_V.parameters(), lr=0.5, max_iter=20,
            history_size=10, line_search_fn='strong_wolfe')

        for step in range(self.cfg.stage3_steps):
            def closure():
                optimizer.zero_grad()
                S = self.S_flat.requires_grad_(True)
                t = self.t_flat.requires_grad_(True)
                V = self.mlp_V(S, t)
                phi = self.mlp_phi(S, t)
                LV = self.loss_fn.compute_bs_residual(V, S, t)
                L_bs, L_ex, L_int = self.loss_fn.energy(V, phi, LV, S, t)
                L_bc = self._bc_loss()
                loss = 10 * (L_bs + L_ex + L_int) + self.cfg.lambda_bc * L_bc
                loss.backward()
                nn.utils.clip_grad_norm_(self.mlp_V.parameters(), self.cfg.grad_clip)
                return loss
            loss = optimizer.step(closure)
            if (step + 1) % 5 == 0:
                print(f"  L-BFGS Step {step+1}/{self.cfg.stage3_steps} | Loss={loss.item():.6f}")

    def train(self):
        self.train_stage1()
        self.train_stage2()
        self.train_stage3()
        self.save_results()

    def save_results(self):
        self.mlp_V.eval()
        self.mlp_phi.eval()

        with torch.no_grad():
            S_test = torch.linspace(0, self.cfg.S_max, 200, device=self.device)
            t_test = torch.linspace(0, self.cfg.T, 100, device=self.device)
            Sm, tm = torch.meshgrid(S_test, t_test, indexing='ij')
            V_test = self.mlp_V(Sm.reshape(-1, 1), tm.reshape(-1, 1)).reshape(200, 100)
            phi_test = self.mlp_phi(Sm.reshape(-1, 1), tm.reshape(-1, 1)).reshape(200, 100)

        np.savez('scheme2_results.npz',
                 S=S_test.cpu().numpy(), t=t_test.cpu().numpy(),
                 V=V_test.cpu().numpy(), phi=phi_test.cpu().numpy())
        with open('scheme2_history.json', 'w') as f:
            json.dump(self.history, f)
        print(f"\n结果已保存: scheme2_results.npz, scheme2_history.json")


if __name__ == "__main__":
    print("方案2: MLP学φ + MLP学V + Phase Field能量泛函")
    print(f"设备: {config.device}")
    trainer = Scheme2Trainer(config)
    trainer.train()
