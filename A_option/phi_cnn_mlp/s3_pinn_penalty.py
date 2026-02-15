# ================================================================
# scheme3_pinn_penalty.py
# 方案3: MLP学V + 惩罚法 (传统PINN基线, 无φ)
# ================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json, time, warnings
warnings.filterwarnings('ignore')

from scheme1_cnn_phasefield import (
    Config, generate_fdm_solution, get_fdm_on_grid, SirenLayer, ValueMLP
)

config = Config()
torch.manual_seed(config.seed)
np.random.seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)


# ================================================================
# 惩罚法损失
# ================================================================
class PenaltyLoss:
    def __init__(self, cfg):
        self.K = cfg.K
        self.r = cfg.r
        self.sigma = cfg.sigma

    def compute_bs_residual(self, V, S, t):
        V_S = torch.autograd.grad(V, S, torch.ones_like(V),
                                  create_graph=True, retain_graph=True)[0]
        V_SS = torch.autograd.grad(V_S, S, torch.ones_like(V_S),
                                   create_graph=True, retain_graph=True)[0]
        V_t = torch.autograd.grad(V, t, torch.ones_like(V),
                                  create_graph=True, retain_graph=True)[0]
        return V_t + 0.5 * self.sigma**2 * S**2 * V_SS + self.r * S * V_S - self.r * V

    def penalty_loss(self, V, LV, S, penalty_lambda):
        """
        传统PINN惩罚法:
        L = ||LV||² + λ||max(Ψ-V, 0)||²
        """
        payoff = torch.clamp(self.K - S, min=0)
        L_pde = torch.mean(LV**2)
        violation = torch.clamp(payoff - V, min=0)
        L_penalty = penalty_lambda * torch.mean(violation**2)
        return L_pde, L_penalty


# ================================================================
# 训练器
# ================================================================
class Scheme3Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        self.mlp = ValueMLP(cfg.mlp_hidden, cfg.mlp_layers, cfg.omega_0).to(self.device)
        self.loss_fn = PenaltyLoss(cfg)

        S_g = torch.linspace(0, cfg.S_max, cfg.grid_S, device=self.device)
        t_g = torch.linspace(0, cfg.T, cfg.grid_t, device=self.device)
        Sm, tm = torch.meshgrid(S_g, t_g, indexing='ij')
        self.S_flat = Sm.reshape(-1, 1)
        self.t_flat = tm.reshape(-1, 1)
        self.S_mesh = Sm
        self.t_mesh = tm

        _, _, V_fdm = get_fdm_on_grid(cfg)
        self.V_fdm = torch.from_numpy(V_fdm).float().to(self.device)

        self._prepare_bc()
        self.history = {'loss': [], 'pde': [], 'arb': [], 'time': []}

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
        return F.mse_loss(self.mlp(S, t), self.bc_V)

    def train_stage1(self):
        print("=" * 70)
        print("Stage 1: MLP预热 (FDM监督)")
        print("=" * 70)

        optimizer = AdamW(self.mlp.parameters(), lr=self.cfg.lr_mlp * 2, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.stage1_epochs, eta_min=1e-6)

        for epoch in range(self.cfg.stage1_epochs):
            self.mlp.train()
            optimizer.zero_grad()
            V_pred = self.mlp(self.S_flat, self.t_flat).reshape(self.cfg.grid_S, self.cfg.grid_t)
            loss = F.mse_loss(V_pred, self.V_fdm) + self.cfg.lambda_bc * self._bc_loss()
            loss.backward()
            nn.utils.clip_grad_norm_(self.mlp.parameters(), self.cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 500 == 0:
                rel = torch.mean(torch.abs(V_pred - self.V_fdm) / (self.V_fdm + 1e-6)).item()
                print(f"  Epoch {epoch+1}/{self.cfg.stage1_epochs} | Loss={loss.item():.6f} | RelErr={rel:.4f}")

    def train_stage2(self):
        print("\n" + "=" * 70)
        print("Stage 2: PINN + 惩罚法训练")
        print("=" * 70)

        optimizer = AdamW(self.mlp.parameters(), lr=self.cfg.lr_mlp, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.stage2_epochs, eta_min=1e-6)

        best_loss = float('inf')
        t0 = time.time()

        for epoch in range(self.cfg.stage2_epochs):
            self.mlp.train()
            optimizer.zero_grad()

            S = self.S_flat.requires_grad_(True)
            t = self.t_flat.requires_grad_(True)
            V = self.mlp(S, t)
            LV = self.loss_fn.compute_bs_residual(V, S, t)

            # 惩罚系数逐步增大 (类比ε→0)
            if epoch < 3000:
                pen_lambda = 100.0
            elif epoch < 8000:
                pen_lambda = 500.0
            else:
                pen_lambda = 1000.0

            L_pde, L_pen = self.loss_fn.penalty_loss(V, LV, S, pen_lambda)
            L_bc = self._bc_loss()
            loss = L_pde + L_pen + self.cfg.lambda_bc * L_bc

            loss.backward()
            nn.utils.clip_grad_norm_(self.mlp.parameters(), self.cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                V_g = V.reshape(self.cfg.grid_S, self.cfg.grid_t)
                payoff = torch.clamp(self.cfg.K - self.S_mesh, min=0)
                arb = torch.max(torch.clamp(payoff - V_g, min=0)).item()

            self.history['loss'].append(loss.item())
            self.history['pde'].append(L_pde.item())
            self.history['arb'].append(arb)
            self.history['time'].append(time.time() - t0)

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.mlp.state_dict(), 'scheme3_best.pth')

            if (epoch + 1) % 1000 == 0:
                print(f"  Epoch {epoch+1}/{self.cfg.stage2_epochs} | "
                      f"PDE={L_pde.item():.4f} | Pen={L_pen.item():.4f} | "
                      f"Arb={arb:.4f} | λ={pen_lambda}")

        self.mlp.load_state_dict(torch.load('scheme3_best.pth', weights_only=False))

    def train_stage3(self):
        print("\n" + "=" * 70)
        print("Stage 3: L-BFGS精调")
        print("=" * 70)

        optimizer = torch.optim.LBFGS(
            self.mlp.parameters(), lr=0.5, max_iter=20,
            history_size=10, line_search_fn='strong_wolfe')

        for step in range(self.cfg.stage3_steps):
            def closure():
                optimizer.zero_grad()
                S = self.S_flat.requires_grad_(True)
                t = self.t_flat.requires_grad_(True)
                V = self.mlp(S, t)
                LV = self.loss_fn.compute_bs_residual(V, S, t)
                L_pde, L_pen = self.loss_fn.penalty_loss(V, LV, S, 1000.0)
                L_bc = self._bc_loss()
                loss = L_pde + L_pen + self.cfg.lambda_bc * L_bc
                loss.backward()
                nn.utils.clip_grad_norm_(self.mlp.parameters(), self.cfg.grad_clip)
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
        self.mlp.eval()
        with torch.no_grad():
            S_test = torch.linspace(0, self.cfg.S_max, 200, device=self.device)
            t_test = torch.linspace(0, self.cfg.T, 100, device=self.device)
            Sm, tm = torch.meshgrid(S_test, t_test, indexing='ij')
            V_test = self.mlp(Sm.reshape(-1, 1), tm.reshape(-1, 1)).reshape(200, 100)

        np.savez('scheme3_results.npz',
                 S=S_test.cpu().numpy(), t=t_test.cpu().numpy(),
                 V=V_test.cpu().numpy())
        with open('scheme3_history.json', 'w') as f:
            json.dump(self.history, f)
        print(f"\n结果已保存: scheme3_results.npz, scheme3_history.json")


if __name__ == "__main__":
    print("方案3: MLP学V + 惩罚法 (传统PINN基线)")
    print(f"设备: {config.device}")
    trainer = Scheme3Trainer(config)
    trainer.train()
