"""Semi-parametric marginals via peaks-over-threshold.

For each feature and each tail the marginal CDF is

    F(x) = q_lo * GPD_sf(u_lo - x; xi_lo, beta_lo)          for x <  u_lo
    F(x) = empirical (interpolated)                          for u_lo <= x <= u_hi
    F(x) = 1 - q_hi * GPD_sf(x - u_hi; xi_hi, beta_hi)       for x >  u_hi

with GPD_sf(y; xi, beta) = (1 + xi y / beta)^(-1/xi), justified by
Pickands-Balkema-de Haan.  The PIT z = T_nu^{-1}(F(x)) then makes every marginal
exactly t_nu, matching the flow-matching base so the flow only has to transport the
copula.

The two tails are kept separate throughout -- a return series has no reason for its
two tails to be equally heavy -- and the body is pinned to F(u_lo) = q_lo and
F(u_hi) = 1 - q_hi so the piecewise CDF is continuous.

The GPD parameters fitted here are refined afterwards by evt_shrink.shrink_ensemble,
which pools xi across features and imposes xi >= 0; see that module.

Conventions: 'lower'/'upper' refer to tails of the raw variable.  Risk of losses
L = -r corresponds to the lower tail of returns.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

_EPS = 1e-12


def hill_estimator(x: np.ndarray, k_frac: float = 0.05, tail: str = "lower") -> float:
    """Hill estimate of the tail index alpha (heavier tail <=> smaller alpha),

        alpha_hat^{-1} = (1/k) sum_{i=1..k} log( X_(n-i+1) / X_(n-k) ),

    on the positive exceedances of the requested tail (x -> -x for 'lower').  k is
    k_frac of the FULL sample, not of the positive part, so k_frac matches the
    threshold quantile used elsewhere.  Returns inf if the tail is too thin.
    """
    z = np.asarray(x, dtype=float)
    y = -z if tail == "lower" else z
    k = max(10, int(k_frac * y.size))
    y = np.sort(y[y > 0.0])
    if y.size < k + 1:
        return np.inf
    inv_alpha = np.mean(np.log(y[-k:] / y[-k - 1]))
    return 1.0 / max(inv_alpha, _EPS)


def fit_gpd(exc: np.ndarray) -> tuple[float, float]:
    """MLE of (xi, beta) for exceedances over a threshold, location fixed at 0.

    Raises on a degenerate exceedance set -- the threshold coinciding with the sample
    extremum, e.g. an accrual series that never falls.
    """
    y = np.asarray(exc, dtype=float)
    y = y[np.isfinite(y) & (y > 0.0)]
    n = y.size
    if n < 5:
        raise ValueError(
            f"GPD fit needs >=5 positive exceedances, got {n}; the threshold "
            "coincides with the sample extremum (a point mass there, e.g. an "
            "accrual series that never falls)")
    xi, _, beta = stats.genpareto.fit(y, floc=0.0)
    if not (np.isfinite(xi) and np.isfinite(beta) and beta > 0.0):
        raise ValueError(f"GPD MLE did not converge on {n} exceedances")
    return float(xi), float(max(beta, _EPS))


class SemiParametricMarginal:
    """Marginal model for one feature: empirical body + GPD tails + t_nu PIT."""

    def __init__(self, q_tail: float = 0.05, nu: float = 5.0):
        assert 0.0 < float(q_tail) < 0.5
        self.q_tail, self.nu = float(q_tail), float(nu)

    def fit(self, x: np.ndarray) -> "SemiParametricMarginal":
        x = np.sort(np.asarray(x, dtype=float).ravel())
        self.n_ = x.size
        q = self.q_tail
        u_lo, u_hi = np.quantile(x, q), np.quantile(x, 1.0 - q)
        e_lo, e_hi = u_lo - x[x < u_lo], x[x > u_hi] - u_hi
        self.q_lo_, self.u_lo_ = q, float(u_lo)
        self.q_hi_, self.u_hi_ = q, float(u_hi)
        self.xi_lo_, self.beta_lo_ = fit_gpd(e_lo)
        self.xi_hi_, self.beta_hi_ = fit_gpd(e_hi)
        self.n_exc_lo_, self.n_exc_hi_ = int(e_lo.size), int(e_hi.size)

        # Interpolated empirical body on plotting positions (i - 0.5)/n, pinned to the
        # thresholds so the piecewise CDF is continuous.
        p = (np.arange(1, self.n_ + 1) - 0.5) / self.n_
        mask = (p > self.q_lo_) & (p < 1.0 - self.q_hi_)
        xs = np.concatenate([[self.u_lo_], x[mask], [self.u_hi_]])
        ps = np.concatenate([[self.q_lo_], p[mask], [1.0 - self.q_hi_]])
        xs, idx = np.unique(xs, return_index=True)   # strictly increasing for interp
        self._body_x, self._body_p = xs, ps[idx]
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.interp(x, self._body_x, self._body_p)
        lo, hi = x < self.u_lo_, x > self.u_hi_
        if lo.any():
            out[lo] = self.q_lo_ * stats.genpareto.sf(
                self.u_lo_ - x[lo], c=self.xi_lo_, scale=self.beta_lo_)
        if hi.any():
            out[hi] = 1.0 - self.q_hi_ * stats.genpareto.sf(
                x[hi] - self.u_hi_, c=self.xi_hi_, scale=self.beta_hi_)
        return np.clip(out, _EPS, 1.0 - _EPS)

    def ppf(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
        out = np.interp(p, self._body_p, self._body_x)
        lo, hi = p < self.q_lo_, p > 1.0 - self.q_hi_
        if lo.any():
            out[lo] = self.u_lo_ - stats.genpareto.isf(
                p[lo] / self.q_lo_, c=self.xi_lo_, scale=self.beta_lo_)
        if hi.any():
            out[hi] = self.u_hi_ + stats.genpareto.isf(
                (1.0 - p[hi]) / self.q_hi_, c=self.xi_hi_, scale=self.beta_hi_)
        return out

    def transform(self, x: np.ndarray) -> np.ndarray:
        """x -> z with z ~ t_nu marginally."""
        return stats.t.ppf(self.cdf(x), df=self.nu)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return self.ppf(stats.t.cdf(np.asarray(z, dtype=float), df=self.nu))


class MarginalEnsemble:
    """Per-feature SemiParametricMarginal for arrays shaped (..., f)."""

    def __init__(self, q_tail: float = 0.05, nu: float = 5.0):
        self.q_tail, self.nu = float(q_tail), float(nu)

    def fit(self, x: np.ndarray) -> "MarginalEnsemble":
        x = np.asarray(x, dtype=float)
        f = x.shape[-1]
        cols = x.reshape(-1, f)
        self.nu_ = self.nu
        self.marginals_ = []
        for j in range(f):
            try:
                self.marginals_.append(
                    SemiParametricMarginal(self.q_tail, self.nu_).fit(cols[:, j]))
            except ValueError as e:      # name the column instead of a bare traceback
                raise ValueError(f"feature index {j}: {e}") from None
        return self

    def summary(self) -> dict:
        m = self.marginals_
        return dict(
            q_lo=np.array([x.q_lo_ for x in m]),
            q_hi=np.array([x.q_hi_ for x in m]),
            xi_lo=np.array([x.xi_lo_ for x in m]),
            xi_hi=np.array([x.xi_hi_ for x in m]),
            n_exc_lo=np.array([x.n_exc_lo_ for x in m]),
            n_exc_hi=np.array([x.n_exc_hi_ for x in m]),
        )

    def _apply(self, x: np.ndarray, method: str) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        shape, f = x.shape, x.shape[-1]
        flat = x.reshape(-1, f)
        out = np.stack([getattr(self.marginals_[j], method)(flat[:, j])
                        for j in range(f)], axis=-1)
        return out.reshape(shape)

    def transform(self, x):          return self._apply(x, "transform")
    def inverse_transform(self, z):  return self._apply(z, "inverse_transform")
