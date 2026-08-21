"""Conditional Flow Matching with the linear (OT) path and a heavy-tailed base.

Path and target velocity under an independent coupling (x0, x1) ~ p0 x p1:

    x_t = (1 - t) x_0 + t x_1,      u_t(x_t | x_0, x_1) = x_1 - x_0,

    L(theta) = E_{t ~ U[0,1], x0 ~ p0, x1 ~ p_data} || v_theta(x_t, t) - (x_1 - x_0) ||^2.

The L2-minimiser is v*(x, t) = E[x_1 - x_0 | x_t = x], whose ODE flow transports p0 to
p_data (Lipman et al. 2023; Tong et al. 2023 for arbitrary sources).  Nothing requires
a Gaussian source, which is what licenses the Student-t base of base.py.

Sampling integrates dx/dt = v_theta(x, t) from 0 to 1 with Heun's method.
"""

from __future__ import annotations

import copy

import torch

from .base import sample_base
from .model import VelocityField


class EMA:
    """Exponential moving average: shadow <- d * shadow + (1-d) * param."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ps, pm in zip(self.shadow.parameters(), model.parameters()):
            ps.lerp_(pm.detach(), 1.0 - self.decay)
        for bs, bm in zip(self.shadow.buffers(), model.buffers()):
            bs.copy_(bm)


def train_cfm(model: VelocityField, data: torch.Tensor, nu: float,
              steps: int = 20_000, batch_size: int = 128, lr: float = 3e-4,
              weight_decay: float = 1e-4, mix_dim: str = "window",
              device: str = "cpu", ema_decay: float = 0.999,
              log_every: int = 200, seed: int = 0) -> tuple[EMA, list[float]]:
    """Train v_theta on windows `data` of shape (N, n, f), in z-space."""
    torch.manual_seed(seed)
    model = model.to(device).train()
    data = data.to(device)
    N, n, f = data.shape
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    ema, losses = EMA(model, ema_decay), []

    for step in range(1, steps + 1):
        x1 = data[torch.randint(0, N, (batch_size,), device=device)]
        x0 = sample_base(batch_size, n, f, nu, mix_dim=mix_dim, device=device)
        t = torch.rand(batch_size, 1, 1, device=device)
        xt = (1.0 - t) * x0 + t * x1
        loss = ((model(xt, t) - (x1 - x0)) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); ema.update(model)

        losses.append(loss.item())
        if step % log_every == 0 or step == 1:
            avg = sum(losses[-log_every:]) / min(log_every, len(losses))
            print(f"  step {step:>6d}/{steps}  loss {avg:.4f}")
    return ema, losses


@torch.no_grad()
def sample(model: VelocityField, num: int, n: int, f: int, nu: float,
           n_steps: int = 100, mix_dim: str = "window", device: str = "cpu",
           batch_size: int = 512, seed: int = 0) -> torch.Tensor:
    """Draw `num` windows by Heun integration of dx/dt = v_theta(x, t), t: 0 -> 1."""
    torch.manual_seed(seed)
    model = model.to(device).eval()
    out = []
    for start in range(0, num, batch_size):
        b = min(batch_size, num - start)
        x = sample_base(b, n, f, nu, mix_dim=mix_dim, device=device)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t0 = torch.full((b, 1, 1), k * dt, device=device)
            v0 = model(x, t0)
            if k == n_steps - 1:            # terminal Euler step, avoids t = 1
                x = x + dt * v0
            else:
                v1 = model(x + dt * v0, t0 + dt)
                x = x + 0.5 * dt * (v0 + v1)
        out.append(x.cpu())
    return torch.cat(out, dim=0)
