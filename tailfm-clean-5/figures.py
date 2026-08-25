"""Diagnostic figures for real vs. generated windows -- one PNG per diagnostic.

Shared by fit_returns.py (a single generator) and run_baselines.py (several), so both
emit the same files.  Each function takes an ordered mapping gens = {label: (M, n, f)}
and compares every entry with the same real windows (always black, lw=2); colours
follow the insertion order of `gens`.

    qq_lower_tail.png            lower-tail QQ, per feature + pooled
    tail_dependence.png          lambda_L(q), worst `max_pairs` feature pairs
    portfolio_loss_survival.png  P(L > l) of the h-step portfolio loss
    empirical_distributions.png  per feature: log-density and 1-step loss survival

Pair and panel counts are capped: at f = 235 there are 27495 pairs, and one panel per
feature asks matplotlib for a canvas past its 2**16 px limit.  Pairs are ranked by
|lambda_gen(q0) - lambda_real(q0)| at q0 = screen_q (one f x f indicator product per
sample, so O(N f^2) once rather than per pair) and the per-feature grids show the
`max_features` features with the heaviest empirical lower tail.

That ranking maxes over models, so one saturated generator picks the panels for all of
them; `pair_select` in tail_dependence_figure / save_all_figures switches to a random
or lambda_real-stratified selection instead.  See tail_dependence_figure.
"""

from __future__ import annotations

import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tailfm import portfolio_losses

REAL_KW = dict(color="k", lw=2)                 # the real sample, everywhere
MAX_PAIRS = 30                                  # tail-dependence panels, worst-first
MAX_ROWS = 200_000                              # rows used for the rank statistics
MAX_FEATURES = 12                               # per-feature panels (qq, densities)

# PANEL COUNT.  qq_figure and empirical_distribution_figure allocate one panel per
# feature.  At f = 235 that is a 2058 x 47558 px canvas for the QQ grid and a
# 177660 x 1148 px canvas for the densities -- the latter exceeds matplotlib's
# hard 2**16 px limit and raises, killing the run after training and generation
# have already been paid for.  Both now show only the `max_features` features with
# the heaviest empirical lower tail (largest 1% loss quantile), which are the ones
# a tail model is judged on; max_features=None restores one panel per feature.


def _worst_features(real, k):
    """Indices of the k features with the most extreme empirical 1% loss."""
    R = real.reshape(-1, real.shape[-1])
    if k is None or R.shape[1] <= k:
        return list(range(R.shape[1]))
    return sorted(np.argsort(-np.quantile(-R, 0.99, axis=0))[:k].tolist())


def model_colors(gens: dict) -> dict:
    return {name: f"C{i}" for i, name in enumerate(gens)}


def _grid(k: int, max_cols: int = 3) -> tuple[int, int]:
    """(nrow, ncol) for k panels, at most `max_cols` per row."""
    ncol = min(max_cols, max(k, 1))
    return math.ceil(k / ncol), ncol


def _survival(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted sample and its empirical survival function 1 - i/(N+1)."""
    xs = np.sort(np.asarray(x, dtype=float))
    return xs, 1.0 - np.arange(1, xs.size + 1) / (xs.size + 1.0)


def _blank(axes, used: int) -> None:
    for ax in axes.ravel()[used:]:
        ax.axis("off")


# --------------------------------------------------------- pair ranking
def _worst(scores: np.ndarray, keys: list, max_keep: int | None) -> list:
    """Keys of the `max_keep` largest scores, descending."""
    scores = np.asarray(scores, dtype=float)
    if max_keep is None or len(keys) <= max_keep:
        return [keys[i] for i in np.argsort(-scores)]
    idx = np.argpartition(-scores, max_keep)[:max_keep]
    return [keys[i] for i in idx[np.argsort(-scores[idx])]]


def _subsample(x: np.ndarray, n_max: int, seed: int = 0) -> np.ndarray:
    if x.shape[0] <= n_max:
        return x
    rng = np.random.default_rng(seed)
    return x[np.sort(rng.choice(x.shape[0], n_max, replace=False))]


def _uniform_scores(x: np.ndarray) -> np.ndarray:
    """(N, f) -> rank/(N+1) pseudo-observations, column by column to cap memory."""
    N, f = x.shape
    u = np.empty((N, f), dtype=np.float32)
    for j in range(f):
        u[:, j] = (np.argsort(np.argsort(x[:, j])) + 1.0) / (N + 1.0)
    return u


def _lambda_matrix(u: np.ndarray, q0: float, tail: str) -> np.ndarray:
    """f x f matrix of lambda_hat(q0) from one indicator matrix product."""
    v = u if tail == "lower" else 1.0 - u
    b = (v < q0).astype(np.float32)
    return (b.T @ b) / (q0 * b.shape[0])


def _lambda_curve(u: np.ndarray, i: int, j: int, q_grid, tail: str) -> np.ndarray:
    """lambda(q) for one pair, from cached pseudo-observations."""
    ui, uj = (u[:, i], u[:, j]) if tail == "lower" else (1.0 - u[:, i], 1.0 - u[:, j])
    return np.array([np.mean((ui < q) & (uj < q)) / q for q in q_grid])


# --------------------------------------------------------------------- QQ
def qq_figure(real, gens, names, path, q_lo=0.001, q_hi=0.05, n_q=150,
              max_features: int | None = MAX_FEATURES):
    """Lower-tail QQ plots: one panel per feature, plus one with features pooled.

    Points below the 45-degree line mean the generated lower quantile is more
    negative than the real one, i.e. the generated left tail is too heavy.
    """
    f = real.shape[-1]
    colors = model_colors(gens)
    R = real.reshape(-1, f)
    G = {name: gen.reshape(-1, f) for name, gen in gens.items()}
    ql = np.linspace(q_lo, q_hi, n_q)

    sel = _worst_features(real, max_features)
    panels = [(names[j], R[:, j], {k: g[:, j] for k, g in G.items()}) for j in sel]
    if f > 1:
        # Pooled panel on a subsample: np.quantile over the full (M*n, f) array
        # sorts 2.3 GB at --gen 50000 and adds nothing the subsample does not.
        panels.append(("pooled features", _subsample(R, MAX_ROWS),
                       {k: _subsample(g, MAX_ROWS) for k, g in G.items()}))

    nrow, ncol = _grid(len(panels))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 4.3 * nrow),
                             squeeze=False)
    for ax, (title, r, gs) in zip(axes.ravel(), panels):
        rq = np.quantile(r, ql)
        lo = float(rq.min())
        for name, g in gs.items():
            gq = np.quantile(g, ql)
            ax.plot(rq, gq, ".", ms=3, color=colors[name], label=name)
            lo = min(lo, float(gq.min()))
        ax.plot([lo, 0.0], [lo, 0.0], "k--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("real quantile")
        ax.set_ylabel("generated quantile")
        ax.legend(fontsize=7)
    _blank(axes, len(panels))
    fig.suptitle(f"Lower-tail QQ, q in [{q_lo:.1%}, {q_hi:.1%}] "
                 "(below the diagonal = generated tail too heavy)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------- tail dependence
def tail_dependence_figure(real, gens, names, path, q_grid=None, tail="lower",
                           max_pairs: int | None = MAX_PAIRS,
                           screen_q: float = 0.05, max_rows: int = MAX_ROWS,
                           pair_select: str = "worst", seed: int = 0):
    """lambda_L(q) = P(U_i < q, U_j < q) / q for `max_pairs` selected pairs.

    Pairs are screened at a single q with one f x f indicator product; only the
    survivors get full curves.  `pair_select` decides which ones survive:

        worst    largest |lambda_gen(q0) - lambda_real(q0)|, maxed over models.
        random   a uniform sample of the f(f-1)/2 pairs.
        spread   evenly spaced in lambda_real(q0), so the panels cover the whole
                 range of real tail dependence rather than one end of it.

    `worst` maximises the discrepancy OVER MODELS, so a single saturated
    generator selects the panels for all of them: a mode-collapsed baseline emits
    near-comonotone windows, lambda_gen(q) == 1 for almost every pair, and the
    ranking then returns the least tail-dependent REAL pairs -- every panel shows
    the same flat pair of lines.  Use `spread` whenever one model may be
    degenerate, which is the normal case in run_baselines.py.
    """
    f = real.shape[-1]
    if f < 2:
        return None
    if q_grid is None:
        q_grid = np.linspace(0.005, 0.10, 20)
    colors = model_colors(gens)

    # Rank once per sample and reuse: tail_dependence_report re-ranks the whole
    # (N, f) array inside every tail_dependence_curve call, once per pair.
    uR = _uniform_scores(_subsample(real.reshape(-1, f), max_rows))
    uG = {name: _uniform_scores(_subsample(gen.reshape(-1, f), max_rows))
          for name, gen in gens.items()}

    lamR = _lambda_matrix(uR, screen_q, tail)
    D = np.max([np.abs(_lambda_matrix(u, screen_q, tail) - lamR)
                for u in uG.values()], axis=0)
    iu = np.triu_indices(f, k=1)
    keys = list(zip(*iu))
    n_keep = len(keys) if max_pairs is None else min(max_pairs, len(keys))
    if pair_select == "worst":
        pairs = _worst(D[iu], keys, max_pairs)
    elif pair_select == "random":
        sel = np.sort(np.random.default_rng(seed).choice(len(keys), n_keep,
                                                         replace=False))
        pairs = [keys[i] for i in sel]
    elif pair_select == "spread":            # evenly spaced in lambda_real(q0)
        order = np.argsort(lamR[iu])
        pairs = [keys[i] for i in
                 order[np.linspace(0, len(order) - 1, n_keep).astype(int)]]
    else:
        raise ValueError(f"pair_select must be worst/random/spread, got "
                         f"{pair_select!r}")

    nrow, ncol = _grid(len(pairs))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 4.1 * nrow),
                             squeeze=False)
    for ax, (i, j) in zip(axes.ravel(), pairs):
        ax.plot(q_grid, _lambda_curve(uR, i, j, q_grid, tail),
                label="real", **REAL_KW)
        for name, u in uG.items():
            ax.plot(q_grid, _lambda_curve(u, i, j, q_grid, tail), "--",
                    color=colors[name], label=name)
        ax.set_ylim(0, 1)
        ax.set_title(f"({names[i]}, {names[j]})  d={D[i, j]:.3f}", fontsize=9)
        ax.set_xlabel("q")
        ax.set_ylabel(r"$\hat\lambda_L(q)$")
        ax.legend(fontsize=7)
    _blank(axes, len(pairs))
    n_all = f * (f - 1) // 2
    shown = "" if max_pairs is None or n_all <= max_pairs else \
        f" -- {pair_select} {len(pairs)} of {n_all} pairs"
    fig.suptitle(rf"{tail.capitalize()} tail dependence $\hat\lambda(q)$ "
                 "per feature pair (plateau > 0 = co-crash, decay to 0 = tail "
                 f"independence){shown}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ------------------------------------------------------ portfolio survival
def portfolio_survival_figure(real, gens, path, weights=None, horizon=10,
                              real_label="real (train)"):
    """Survival function of the h-step portfolio loss, log y-axis."""
    colors = model_colors(gens)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    L, sf = _survival(portfolio_losses(real, weights=weights, horizon=horizon))
    ax.semilogy(L, sf, label=real_label, **REAL_KW)
    for name, gen in gens.items():
        L, sf = _survival(portfolio_losses(gen, weights=weights, horizon=horizon))
        ax.semilogy(L, sf, color=colors[name], label=name)
    ax.set_title(f"Portfolio loss survival (h={horizon})")
    ax.set_xlabel("loss l")
    ax.set_ylabel("P(L > l)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# -------------------------------------------- per-feature marginal picture
def empirical_distribution_figure(real, gens, names, path,
                                  max_features: int | None = MAX_FEATURES):
    """Per feature: pooled 1-step density (log y) and loss survival P(-X > x)."""
    f = real.shape[-1]
    colors = model_colors(gens)
    R = real.reshape(-1, f)
    G = {name: gen.reshape(-1, f) for name, gen in gens.items()}

    sel = _worst_features(real, max_features)
    fig, axes = plt.subplots(2, len(sel), figsize=(5.4 * len(sel), 8.2),
                             squeeze=False)
    for col, j in enumerate(sel):
        # row 1: pooled 1-step empirical density on a log scale (both tails)
        ax = axes[0][col]
        lo = min(R[:, j].min(), *[g[:, j].min() for g in G.values()])
        hi = max(R[:, j].max(), *[g[:, j].max() for g in G.values()])
        bins = np.linspace(lo, hi, 120)
        ax.hist(R[:, j], bins=bins, density=True, histtype="step",
                label="real", **REAL_KW)
        for name, g in G.items():
            ax.hist(g[:, j], bins=bins, density=True, histtype="step",
                    color=colors[name], label=name)
        ax.set_yscale("log")
        ax.set_title(f"{names[j]}: empirical density (log scale)")
        ax.set_xlabel("1-step return")
        if col == 0:
            ax.set_ylabel("density")
        ax.legend(fontsize=7)
        # row 2: 1-step loss survival P(-X > x), the lower tail head-on
        ax = axes[1][col]
        Lr, sr = _survival(-R[:, j])
        ax.semilogy(Lr, sr, label="real", **REAL_KW)
        xmax = Lr[-1]
        for name, g in G.items():
            Lg, sg = _survival(-g[:, j])
            ax.semilogy(Lg, sg, color=colors[name], label=name)
            xmax = max(xmax, Lg[-1])
        ax.set_xlim(0, 1.02 * xmax)
        ax.set_ylim(1e-5, 1)
        ax.set_title(f"{names[j]}: loss survival P(-X > x)")
        ax.set_xlabel("loss x")
        if col == 0:
            ax.set_ylabel("P(-X > x)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ driver
def save_all_figures(real, gens, names, outdir, weights=None, horizon=10,
                     prefix: str = "",
                     max_pairs: int | None = MAX_PAIRS,
                     pair_select: str = "worst", seed: int = 0) -> list[str]:
    """Write the four diagnostic PNGs into `outdir`; return the paths written."""
    os.makedirs(outdir, exist_ok=True)
    p = lambda stem: os.path.join(outdir, f"{prefix}{stem}.png")
    paths = [
        qq_figure(real, gens, names, p("qq_lower_tail")),
        tail_dependence_figure(real, gens, names, p("tail_dependence"),
                               max_pairs=max_pairs, pair_select=pair_select,
                               seed=seed),
        portfolio_survival_figure(real, gens, p("portfolio_loss_survival"),
                                  weights=weights, horizon=horizon),
        empirical_distribution_figure(real, gens, names,
                                      p("empirical_distributions")),
    ]
    return [q for q in paths if q is not None]
