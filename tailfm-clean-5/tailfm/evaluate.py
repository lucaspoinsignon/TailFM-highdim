"""Tail-focused evaluation of generated vs. real windows.

Diagnostics, all pooled over windows:

  * Hill tail index per feature and tail -- does the generator reproduce tail
    thickness?
  * Lower/upper tail-dependence curves per pair,
        lambda_L(q) = P(U_i < q, U_j < q) / q,   U = rank / (N+1),
    whose limit as q -> 0 is the tail-dependence coefficient.  A tail-dependent pair
    plateaus at lambda > 0; tail-independent features decay to 0, and the generator
    must match both.
  * Marginal VaR/CVaR (loss = -x, per feature, 1-step) real vs. generated.
  * ACF of returns (should be ~0) and of squared returns (volatility clustering).

Cost drives two choices.  Pseudo-observations are computed once per sample and reused
-- ranking inside the pair loop is O(N f log N) per pair and does not terminate at
f ~ 200.  Pairs are screened with a single f x f indicator product and only the worst
`max_pairs` get full lambda(q) curves.  `max_rows` subsamples the pooled rows used for
rank statistics; 200k leaves ~4000 exceedances at q = 0.02.
"""

from __future__ import annotations

import numpy as np

from .evt import hill_estimator
from .risk import var_cvar_empirical


MAX_ROWS = 200_000
MAX_PAIRS = 200


def _pool(windows: np.ndarray, max_rows: int | None = None,
          seed: int = 0) -> np.ndarray:
    """(M, n, f) -> (M*n, f), optionally subsampled to `max_rows` rows."""
    w = np.asarray(windows, dtype=float)
    w = w.reshape(-1, w.shape[-1])
    if max_rows is not None and w.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        w = w[np.sort(rng.choice(w.shape[0], max_rows, replace=False))]
    return w


def pseudo_obs(x: np.ndarray, chunk: int = 64) -> np.ndarray:
    """(N, f) -> rank/(N+1) pseudo-observations, in column blocks to cap memory.

    A whole-array double argsort on (200k, 237) allocates ~380 MB of int64 twice
    over; 64-column blocks keep that near 100 MB at no meaningful cost in time.
    """
    x = np.asarray(x, dtype=float)
    n, f = x.shape
    u = np.empty((n, f), dtype=np.float32)
    rank = np.arange(1, n + 1, dtype=np.float32) / (n + 1.0)
    for a in range(0, f, chunk):
        b = min(a + chunk, f)
        # Sort along the CONTIGUOUS axis: argsort(axis=0) on a C-ordered (n, f)
        # block walks a row stride of f*8 bytes per comparison and misses cache on
        # every one.  Transposing the block first, then one argsort + a scatter in
        # place of the second argsort, is 2.5x faster and bit-identical.
        blk = np.ascontiguousarray(x[:, a:b].T)          # (cols, n)
        idx = np.argsort(blk, axis=1)
        out = np.empty(blk.shape, dtype=np.float32)
        np.put_along_axis(out, idx, np.broadcast_to(rank, blk.shape), axis=1)
        u[:, a:b] = out.T
    return u


def hill_table(real: np.ndarray, gen: np.ndarray, k_frac: float = 0.02,
               max_rows: int | None = MAX_ROWS) -> dict:
    R, G = _pool(real, max_rows), _pool(gen, max_rows)
    out = {}
    for j in range(R.shape[1]):
        out[j] = {t: (hill_estimator(R[:, j], k_frac, t),
                      hill_estimator(G[:, j], k_frac, t))
                  for t in ("lower", "upper")}
    return out


def tail_dependence_curve(u: np.ndarray, i: int, j: int,
                          q_grid: np.ndarray, tail: str = "lower") -> np.ndarray:
    """Empirical lambda(q) for one pair, from PSEUDO-OBSERVATIONS u: (N, f).

    Takes u rather than the raw data on purpose: ranking is the expensive step and
    it does not depend on the pair, so callers rank once with pseudo_obs() and pass
    the result in.
    """
    ui, uj = (u[:, i], u[:, j]) if tail == "lower" else (1.0 - u[:, i], 1.0 - u[:, j])
    return np.array([np.mean((ui < q) & (uj < q)) / q for q in q_grid])


def tail_dependence_matrix(u: np.ndarray, q: float,
                           tail: str = "lower") -> np.ndarray:
    """All f(f-1)/2 values of lambda_hat(q) at once, as one f x f matrix.

        Lambda(q) = B^T B / (q N),      B_tj = 1{U_tj < q}

    i.e. a single BLAS call instead of a loop over pairs.
    """
    v = u if tail == "lower" else 1.0 - u
    b = (v < q).astype(np.float32)
    return (b.T @ b) / (q * b.shape[0])


def tail_dependence_report(real: np.ndarray, gen: np.ndarray,
                           q_grid: np.ndarray | None = None,
                           tail: str = "lower",
                           max_pairs: int | None = MAX_PAIRS,
                           max_rows: int | None = MAX_ROWS,
                           seed: int = 0) -> dict:
    """{(i, j): (lambda_real(q_grid), lambda_gen(q_grid))} for the worst pairs.

    Pairs are ranked by |lambda_gen(q0) - lambda_real(q0)| at the largest q on the
    grid -- one f x f indicator product per sample -- and only the top `max_pairs`
    get full curves.  max_pairs=None restores every pair, which is O(f^2) curves
    and only tractable for small f.
    """
    if q_grid is None:
        q_grid = np.linspace(0.01, 0.10, 10)
    q_grid = np.asarray(q_grid, dtype=float)
    uR = pseudo_obs(_pool(real, max_rows, seed))
    uG = pseudo_obs(_pool(gen, max_rows, seed))
    f = uR.shape[1]

    q0 = float(q_grid.max())
    D = np.abs(tail_dependence_matrix(uG, q0, tail)
               - tail_dependence_matrix(uR, q0, tail))
    iu = np.triu_indices(f, 1)
    order = np.argsort(-D[iu])
    if max_pairs is not None:
        order = order[:max_pairs]

    out = {"q_grid": q_grid}
    for k in order:
        i, j = int(iu[0][k]), int(iu[1][k])
        out[(i, j)] = (tail_dependence_curve(uR, i, j, q_grid, tail),
                       tail_dependence_curve(uG, i, j, q_grid, tail))
    return out


def marginal_risk_table(real: np.ndarray, gen: np.ndarray,
                        alphas=(0.95, 0.99, 0.995),
                        max_rows: int | None = MAX_ROWS) -> dict:
    """Per-feature 1-step VaR/CVaR of the loss -x, real vs. generated."""
    R, G = _pool(real, max_rows), _pool(gen, max_rows)
    out = {}
    for j in range(R.shape[1]):
        out[j] = {a: (var_cvar_empirical(-R[:, j], a), var_cvar_empirical(-G[:, j], a))
                  for a in alphas}
    return out


def acf(x: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Mean-over-windows autocorrelation of a (M, n) array, lags 1..max_lag."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=1, keepdims=True)
    denom = (x ** 2).sum(axis=1)
    return np.array([((x[:, :-k] * x[:, k:]).sum(axis=1) / (denom + 1e-12)).mean()
                     for k in range(1, max_lag + 1)])


def print_report(real: np.ndarray, gen: np.ndarray, feature_names=None,
                 max_pairs: int = 30, max_rows: int | None = MAX_ROWS) -> None:
    f = real.shape[-1]
    names = feature_names or [f"feat{j}" for j in range(f)]

    # Pool ONCE.  hill_table, tail_dependence_report and marginal_risk_table each
    # called _pool on both samples, so the (M*n, f) array was gathered six times --
    # 376 MB per gather out of a 2.26 GB `gen` at --gen 50000.  _pool is a no-op on
    # an array that is already 2-D, so the pre-pooled arrays pass straight through
    # with max_rows=None.  `real`/`gen` are still needed in 3-D for the ACF.
    R2, G2 = _pool(real, max_rows), _pool(gen, max_rows)

    print("\n=== Hill tail index (smaller = heavier; gen should match real) ===")
    for j, d in hill_table(R2, G2, max_rows=None).items():
        for t in ("lower", "upper"):
            r, g = d[t]
            print(f"  {names[j]:>8s} {t:>5s}:  real {r:6.2f}   gen {g:6.2f}")

    n_pairs = f * (f - 1) // 2
    print(f"\n=== Lower tail dependence lambda_L(q=0.02): worst "
          f"{min(max_pairs, n_pairs)} of {n_pairs} pairs by |gen - real| ===")
    td = tail_dependence_report(R2, G2, q_grid=np.array([0.02]),
                                max_pairs=max_pairs, max_rows=None)
    rows = [(k, v) for k, v in td.items() if k != "q_grid"]
    for (i, j), val in rows:
        print(f"  ({names[i]},{names[j]}):  real {val[0][0]:.3f}   "
              f"gen {val[1][0]:.3f}   diff {val[1][0] - val[0][0]:+.3f}")

    print("\n=== Marginal 1-step VaR / CVaR of loss (-x) ===")
    for j, d in marginal_risk_table(R2, G2, max_rows=None).items():
        for a, ((vr, cr), (vg, cg)) in d.items():
            print(f"  {names[j]:>8s} a={a:5.3f}:  VaR real {vr:8.4f} gen {vg:8.4f}"
                  f"  |  CVaR real {cr:8.4f} gen {cg:8.4f}")

    print("\n=== ACF (lag 1..5), squared series: volatility clustering ===")
    for j in range(f):
        ar = acf(real[:, :, j] ** 2, 5); ag = acf(gen[:, :, j] ** 2, 5)
        print(f"  {names[j]:>8s} sq-ACF real {np.round(ar, 2)}  gen {np.round(ag, 2)}")
