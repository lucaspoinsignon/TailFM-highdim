"""Heavy-tailed source distribution for flow matching.

Multivariate Student-t by normal variance mixing,

    x_0 = z * sqrt(nu / W),   z ~ N(0, I),   W ~ chi^2_nu,

with a single W shared across a group of coordinates.  Sharing the mixing variable
makes the group jointly elliptically t-distributed, so it has strictly positive tail
dependence

    lambda = 2 t_{nu+1}( -sqrt((nu+1)(1-rho)/(1+rho)) ) > 0   for any rho > -1,

whereas the Gaussian has exactly 0.  The flow then only has to modulate joint-extreme
intensity per pair rather than manufacture it.

mix_dim="window" shares W across features and time; "time" shares it across features
only.  Neither produces within-window volatility clustering -- a window-constant scale
cancels under the within-window normalisation of evaluate.acf, and an independent
per-step scale has no persistence.  Clustering comes from the velocity field instead
(see model.pos_std).
"""

from __future__ import annotations

import torch


def sample_base(batch: int, n: int, f: int, nu: float,
                mix_dim: str = "window",
                device: torch.device | str = "cpu",
                generator: torch.Generator | None = None) -> torch.Tensor:
    z = torch.randn(batch, n, f, device=device, generator=generator)
    w_shape = (batch, 1, 1) if mix_dim == "window" else (batch, n, 1)
    # chi^2_nu = Gamma(shape=nu/2, rate=1/2)
    w = torch.distributions.Gamma(nu / 2.0, 0.5).sample(w_shape).to(device)
    return z * torch.sqrt(torch.as_tensor(nu, device=device) / w.clamp_min(1e-8))
