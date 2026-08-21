"""Pooled GPD tail estimation, applied as a post-fit step to a MarginalEnsemble.

`MarginalEnsemble.fit` is run as usual, then `shrink_ensemble(marg, x)` replaces
(xi, beta) on every SemiParametricMarginal in place.  The empirical body is untouched,
so only the two GPD branches of cdf/ppf move and the piecewise CDF stays continuous at
the thresholds.

Three changes, all aimed at the k ~ 60 exceedances available at T ~ 1200, q = 0.05:

1.  Zhang & Stephens (2009) in place of the MLE.  At k = 62 over 3000 replicates it
    removes the MLE's downward bias (-0.04 -> +0.01 at xi = 0.267) and cuts sd by ~7%,
    and it cannot fail to converge.

2.  Empirical-Bayes pooling of xi across features, with an Efron-Morris
    limited-translation cap.  se(xi) = (1+xi)/sqrt(k) ~ 0.17 here, against a
    cross-sectional sd of ~0.20, so most of the apparent heterogeneity is estimation
    noise.  The unrestricted posterior mean over-shrinks the genuinely heaviest
    features, whose own sample maxima then get survival probabilities far below 1/k;
    capping the displacement at c standard errors keeps ~77% of the RMSE gain without
    that.  tau^2 is estimated by moments, so the procedure is self-limiting: if the
    features really do differ, lambda -> 1 and nothing is shrunk.

3.  A shape floor xi >= 0.  A negative xi gives the GPD a finite endpoint that the MLE
    places essentially at the extreme training observation, so the generator can never
    produce a loss worse than the worst one already seen -- the same defect the
    Tail-GAN baseline has by construction.  Held-out records also breach that endpoint,
    which sends the PIT to the clip and yields |z| ~ 1e2-1e4.

beta is re-estimated by profile MLE conditional on the new xi, so the pair stays a
coherent maximiser rather than two estimates from different fits.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

_EPS = 1e-12


def fit_gpd_zs(exc: np.ndarray) -> tuple[float, float]:
    """Zhang & Stephens (2009) estimator of (xi, beta) for POT exceedances.

    Profiles the likelihood in b = xi/beta on a grid anchored on the sample quartile
    and maximum, then averages b over the grid with profile-likelihood weights instead
    of maximising -- which is where the variance reduction comes from.
    """
    y = np.asarray(exc, dtype=float)
    y = np.sort(y[np.isfinite(y) & (y > 0.0)])
    n = y.size
    if n < 5:
        raise ValueError(f"GPD fit needs >=5 positive exceedances, got {n}")

    m = 30 + int(np.sqrt(n))
    bs = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    bs /= 3.0 * y[int(n / 4 + 0.5) - 1]
    bs += 1.0 / y[-1]
    xis = np.log1p(-bs[:, None] * y).mean(axis=1)          # xi(b) along the grid
    L = n * (np.log(-bs / xis) - xis - 1.0)                # profile log-likelihood
    w = 1.0 / np.exp(L - L[:, None]).sum(axis=1)
    w /= w.sum()
    b = float((bs * w).sum())
    xi = float(np.log1p(-b * y).mean())
    return xi, float(max(-xi / b, _EPS))


def beta_profile(exc: np.ndarray, xi: float) -> float:
    """MLE of beta with xi held fixed."""
    y = np.asarray(exc, dtype=float)
    y = y[np.isfinite(y) & (y > 0.0)]
    n = y.size
    if abs(xi) < 1e-8:                                     # xi -> 0: exponential
        return float(max(y.mean(), _EPS))
    lo = _EPS if xi > 0 else (-xi * y.max()) * (1.0 + 1e-9)   # keep the support valid
    return float(optimize.minimize_scalar(
        lambda s_: n * np.log(s_) + (1.0 / xi + 1.0) * np.sum(np.log1p(xi * y / s_)),
        bounds=(lo, 50.0 * np.median(y)), method="bounded").x)


def eb_shrink_xi(xi_hat: np.ndarray, k: np.ndarray,
                 c: float = 1.0) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Empirical-Bayes shrinkage of xi with an Efron-Morris cap of c standard errors.

    s_j^2 = (1 + xi_j)^2 / k_j is Smith's (1987) asymptotic variance;
    tau^2 = max(0, Var_j(xi_hat) - mean_j(s_j^2)) is the heterogeneity surviving after
    estimation noise.  c = inf gives the unrestricted posterior mean.
    """
    xi_hat = np.asarray(xi_hat, dtype=float)
    k = np.maximum(np.asarray(k, dtype=float), 1.0)
    s2 = (1.0 + xi_hat) ** 2 / k
    mu = float(xi_hat.mean())
    tau2 = float(max(0.0, xi_hat.var(ddof=1) - s2.mean()))
    lam = tau2 / (tau2 + s2)
    move = (1.0 - lam) * (xi_hat - mu)
    if np.isfinite(c):
        move = np.sign(move) * np.minimum(np.abs(move), c * (1.0 + xi_hat) / np.sqrt(k))
    return xi_hat - move, lam, mu, tau2


def exceedances(m, x_col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) exceedances of one fitted marginal at its own thresholds."""
    x = np.asarray(x_col, dtype=float).ravel()
    return m.u_lo_ - x[x < m.u_lo_], x[x > m.u_hi_] - m.u_hi_


def shrink_ensemble(marg, x: np.ndarray, c: float = 1.0,
                    xi_min: float | None = 0.0, verbose: bool = True) -> dict:
    """Re-estimate and pool the tails of a fitted MarginalEnsemble, in place.

    x must be the same array passed to marg.fit -- the thresholds are kept, so this
    only changes how the exceedances above them are described.
    """
    cols = np.asarray(x, dtype=float).reshape(-1, np.shape(x)[-1])
    m_list = marg.marginals_
    f = len(m_list)

    exc = [exceedances(m_list[j], cols[:, j]) for j in range(f)]
    xi_lo = np.array([fit_gpd_zs(e[0])[0] for e in exc])
    xi_hi = np.array([fit_gpd_zs(e[1])[0] for e in exc])
    k_lo = np.array([m.n_exc_lo_ for m in m_list], dtype=float)
    k_hi = np.array([m.n_exc_hi_ for m in m_list], dtype=float)

    new_lo, lam_lo, mu_lo, tau2_lo = eb_shrink_xi(xi_lo, k_lo, c)
    new_hi, lam_hi, mu_hi, tau2_hi = eb_shrink_xi(xi_hi, k_hi, c)
    n_floor = 0
    if xi_min is not None:
        n_floor = int((new_lo < xi_min).sum() + (new_hi < xi_min).sum())
        new_lo = np.maximum(new_lo, xi_min)
        new_hi = np.maximum(new_hi, xi_min)

    for j, m in enumerate(m_list):
        m.xi_lo_, m.beta_lo_ = float(new_lo[j]), beta_profile(exc[j][0], new_lo[j])
        m.xi_hi_, m.beta_hi_ = float(new_hi[j]), beta_profile(exc[j][1], new_hi[j])

    if verbose:
        for side, xi_r, xi_s, lam, mu, t2, k in (
                ("lower", xi_lo, new_lo, lam_lo, mu_lo, tau2_lo, k_lo),
                ("upper", xi_hi, new_hi, lam_hi, mu_hi, tau2_hi, k_hi)):
            se = float(np.mean((1.0 + xi_r) ** 2 / k) ** 0.5)
            print(f"  {side}: pooled mu={mu:+.3f}  sd(xi_hat)={xi_r.std(ddof=1):.3f}"
                  f"  mean se={se:.3f}  ->  tau={np.sqrt(t2):.3f}"
                  f"  mean lambda={lam.mean():.2f}"
                  f"  (range {xi_r.min():+.3f},{xi_r.max():+.3f}"
                  f" -> {xi_s.min():+.3f},{xi_s.max():+.3f})")
        if xi_min is not None:
            print(f"  shape floor xi >= {xi_min:+.2f} applied to {n_floor} of "
                  f"{2 * f} tails")
    return dict(xi_lo_raw=xi_lo, xi_hi_raw=xi_hi, xi_lo=new_lo, xi_hi=new_hi,
                lam_lo=lam_lo, lam_hi=lam_hi, mu_lo=mu_lo, mu_hi=mu_hi,
                tau2_lo=tau2_lo, tau2_hi=tau2_hi, n_floor=n_floor)
