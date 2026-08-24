"""Step 5.  Pooled empirical distribution: real vs generated.

    python 05_density.py --data data/returns_clean.csv runs/final
    python 05_density.py --data data/returns_clean.csv runs/final runs/baselines \
        --scale std --panel both

Kernel density estimate of the marginal value distribution, pooling every scalar
over windows, time steps and features -- the `probability_density.png` figure of
the TSGM/TimeGAN visualisation scripts, adapted to log returns.

What it estimates.  With p_j the 1-step law of feature j, the pooled sample is a
draw from the equal-weight mixture

    p_pool(x) = (1/f) sum_j p_j(x),

for the real rows and for the generated rows respectively.  It is a MARGINAL
diagnostic: it is invariant to the copula, so it says nothing about the joint
tail structure the flow is trained to transport (use tail_dependence.png,
06_compare.py, summarize_runs.py for that).  Two further caveats worth carrying:

  * Under the default `--recalibrate` path of fit_returns.py the generated
    marginals are F_hat_j by construction (rank recalibration), so this figure
    scores the EVT MARGINAL FIT, not the flow.  Run the fit with
    --no-recalibrate if you want it to score the generator's own marginals.
  * The pooled scalars are far from independent: f values from one date share a
    cross-section, n dates from one window overlap, and consecutive windows
    overlap at stride 1.  The effective sample size behind the curve is closer
    to the number of independent DATES (~T_train) than to the number of pooled
    scalars, so do not read fine structure into it.

Scaling (`--scale`), applied with statistics taken from the real data only, so
real and generated stay in the same units:

    none    raw log returns (default).  Pooling then weights the high-volatility
            features most, which is what a portfolio-level picture should do.
    std     x / sd_j(train).  Puts every feature on a common scale, so the pooled
            curve is not just the loudest handful of columns.
    minmax  (x - min_j) / (max_j - min_j) on the real train statistics, i.e. the
            convention of the TSGM visualisation script.  Included for
            comparability only: both endpoints are single order statistics, and
            the GPD tails are designed to extrapolate past them, so generated
            values legitimately leave [0, 1].

Bandwidth.  Scott's factor h = sd * m^{-1/5} is driven by sd, which a fat tail
inflates, so the mode is oversmoothed.  `--bw robust` uses Silverman's
rule-of-thumb h = 0.9 * min(sd, IQR/1.34) * m^{-1/5}, which the tails cannot
inflate; `--bw-adjust` scales whichever rule is chosen.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from csvio import load_returns, feature_names_from_csv

REAL_KW = dict(color="k", lw=2)          # same convention as figures.py
MAX_POINTS = 200_000


def make_windows(x: np.ndarray, n: int, stride: int = 1) -> np.ndarray:
    """(T, f) -> (N, n, f).  Duplicated from tailfm.data on purpose: importing
    `tailfm` pulls in torch through tailfm.base, and this script is numpy-only,
    like 03_plot.py / 06_compare.py."""
    idx = np.arange(0, len(x) - n + 1, stride)
    return np.stack([x[i:i + n] for i in idx])


# ------------------------------------------------------------------ pooling
def pool_rows(flat, max_points: int, rng, width: int | None = None) -> np.ndarray:
    """(N, f) array or memmap -> (n_rows, f) subsample, n_rows * width ~ max_points.

    Rows are subsampled rather than individual scalars: with a memmapped
    generated_windows.npy that touches only the rows it keeps, and a row is the
    natural sampling unit anyway (the f values of one date are one cross-section,
    not f independent draws).
    """
    N, f = flat.shape
    n_rows = max(1, int(np.ceil(max_points / (width or f))))
    if n_rows >= N:
        return np.asarray(flat, dtype=float)
    idx = np.sort(rng.choice(N, n_rows, replace=False))
    return np.asarray(flat[idx], dtype=float)


def load_gen_flat(path: str):
    """Directory or .npy path -> memmapped (M*n, f) view of the generated windows."""
    p = path if path.endswith(".npy") else os.path.join(path, "generated_windows.npy")
    if not os.path.exists(p):
        return None, p
    g = np.load(p, mmap_mode="r")
    if g.ndim != 3:
        raise SystemExit(f"{p}: expected (M, n, f), got {g.shape}")
    return g.reshape(-1, g.shape[-1]), p


# ----------------------------------------------------------------- density
def kde_curve(v: np.ndarray, grid: np.ndarray, bw: str, bw_adjust: float):
    """Gaussian KDE of `v` on `grid`.  Returns (density, bandwidth h)."""
    v = np.asarray(v, dtype=float)
    sd = float(v.std(ddof=1))
    m = v.size
    if bw == "robust":
        iqr = float(np.subtract(*np.percentile(v, [75, 25])))
        h = 0.9 * min(sd, iqr / 1.349) * m ** (-0.2)
        factor = bw_adjust * h / sd                  # gaussian_kde: h = factor * sd
        kde = stats.gaussian_kde(v, bw_method=factor)
    else:
        kde = stats.gaussian_kde(v, bw_method=bw)
        kde.set_bandwidth(kde.factor * bw_adjust)
    return kde(grid), float(kde.factor * sd)


def two_sample_stats(r: np.ndarray, g: np.ndarray) -> dict:
    """Distances between the two pooled samples.  Statistics only, no p-values:
    the pooled scalars are dependent, so the null distributions do not apply."""
    return dict(ks=float(stats.ks_2samp(r, g, method="asymp").statistic),
                w1=float(stats.wasserstein_distance(r, g)),
                sd_r=float(r.std(ddof=1)), sd_g=float(g.std(ddof=1)),
                kurt_r=float(stats.kurtosis(r, fisher=False)),
                kurt_g=float(stats.kurtosis(g, fisher=False)))


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Pooled marginal density, real vs generated windows")
    ap.add_argument("--data", required=True, help="CSV or .npy of shape (T, f)")
    ap.add_argument("--prices", action="store_true", help="input is prices")
    ap.add_argument("--out", default=None,
                    help="output PNG (default: <first run>/probability_density.png)")
    ap.add_argument("--n", type=int, default=24, help="window length, as fitted")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--real-source", default="rows", choices=["rows", "windows"],
                    help="'rows' pools the train dates once each; 'windows' pools "
                         "make_windows(train) exactly as fit_returns.py builds it, "
                         "which at stride 1 counts every interior date n times")
    ap.add_argument("--scale", default="none", choices=["none", "std", "minmax"],
                    help="per-feature rescaling, fitted on the real train data")
    ap.add_argument("--features", default=None,
                    help="comma-separated feature names to restrict the pool to")
    ap.add_argument("--max-points", type=int, default=MAX_POINTS)
    ap.add_argument("--clip-q", type=float, default=0.001,
                    help="plot range = [q, 1-q] quantiles of the pooled REAL sample; "
                         "0 uses the full range, which one order statistic then sets")
    ap.add_argument("--bw", default="scott", choices=["scott", "silverman", "robust"])
    ap.add_argument("--bw-adjust", type=float, default=1.0)
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--panel", default="linear", choices=["linear", "log", "both"],
                    help="'log' puts the density on a log y-axis, where the tails "
                         "are visible; 'both' draws the two side by side")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("runs", nargs="+",
                    help="run directories holding generated_windows.npy, or .npy paths")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)

    # ------------------------------------------------------------ real pool
    r = load_returns(a.data, a.prices)
    T, f = r.shape
    names = feature_names_from_csv(a.data, f)
    train = r[:int((1.0 - a.test_frac) * T)]

    cols = list(range(f))
    if a.features:
        want = [s.strip() for s in a.features.split(",")]
        miss = [s for s in want if s not in names]
        if miss:
            raise SystemExit(f"not in {a.data}: {miss}\navailable: {names[:10]} ...")
        cols = [names.index(s) for s in want]

    # Scaling statistics from the real TRAIN data only, so both samples share units.
    if a.scale == "std":
        loc = np.zeros(f)
        scl = np.where(train.std(axis=0) == 0.0, 1.0, train.std(axis=0))
    elif a.scale == "minmax":
        loc = train.min(axis=0)
        rngj = train.max(axis=0) - loc
        scl = np.where(rngj == 0.0, 1.0, rngj)
    else:
        loc, scl = np.zeros(f), np.ones(f)

    prep = lambda rows: (((rows - loc) / scl)[:, cols]).ravel()

    real_flat = (make_windows(train, a.n, a.stride).reshape(-1, f)
                 if a.real_source == "windows" else train)
    vR = prep(pool_rows(real_flat, a.max_points, rng, width=len(cols)))

    # ------------------------------------------------------- generated pools
    gens = {}
    for run in a.runs:
        flat, p = load_gen_flat(run)
        if flat is None:
            print(f"  skipping {run}: no {os.path.basename(p)}")
            continue
        if flat.shape[1] != f:
            raise SystemExit(f"{p}: {flat.shape[1]} features, {a.data} has {f}")
        label = os.path.basename(os.path.normpath(run)).replace(".npy", "")
        gens[label] = prep(pool_rows(flat, a.max_points,
                                     np.random.default_rng(a.seed),
                                     width=len(cols)))
    if not gens:
        raise SystemExit("no generated_windows.npy found in any of the given runs")

    print(f"{a.data}: T={T}, f={f} | train {train.shape} | pooling {len(cols)} "
          f"features | scale={a.scale} | real {vR.size} values")

    # ---------------------------------------------------------------- range
    if a.clip_q > 0:
        lo, hi = np.quantile(vR, [a.clip_q, 1.0 - a.clip_q])
    else:
        lo = min(vR.min(), *[v.min() for v in gens.values()])
        hi = max(vR.max(), *[v.max() for v in gens.values()])
    pad = 0.02 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, a.grid)

    dR, hR = kde_curve(vR, grid, a.bw, a.bw_adjust)
    curves = {}
    for label, v in gens.items():
        curves[label] = kde_curve(v, grid, a.bw, a.bw_adjust)[0]
        s = two_sample_stats(vR, v)
        print(f"  {label:>16}: {v.size} values  KS {s['ks']:.4f}  W1 {s['w1']:.3e}  "
              f"sd {s['sd_g']:.4f} (real {s['sd_r']:.4f})  "
              f"kurt {s['kurt_g']:7.2f} (real {s['kurt_r']:7.2f})")
    print(f"  bandwidth h={hR:.3e} ({a.bw}, x{a.bw_adjust})  "
          f"grid [{grid[0]:.4f}, {grid[-1]:.4f}]")

    # --------------------------------------------------------------- figure
    panels = ["linear", "log"] if a.panel == "both" else [a.panel]
    colors = {label: f"C{i}" for i, label in enumerate(curves)}
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 4.6),
                             squeeze=False)
    for ax, kind in zip(axes[0], panels):
        ax.fill_between(grid, dR, alpha=0.25, color="k")
        ax.plot(grid, dR, label="real", **REAL_KW)
        for label, d in curves.items():
            ax.fill_between(grid, d, alpha=0.20, color=colors[label])
            ax.plot(grid, d, color=colors[label], lw=1.5, label=label)
        if kind == "log":
            ax.set_yscale("log")
            floor = max(1e-8, 1e-5 * float(dR.max()))
            ax.set_ylim(floor, 2.0 * max([dR.max()] + [d.max() for d in curves.values()]))
        else:
            ax.set_ylim(bottom=0.0)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_xlabel({"none": "log return", "std": "return / sd(train)",
                       "minmax": "min-max scaled value"}[a.scale])
        ax.set_ylabel("probability density")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    tail = "" if a.clip_q <= 0 else f", x clipped to the real [{a.clip_q:.1%}, " \
                                    f"{1 - a.clip_q:.1%}] quantiles"
    fig.suptitle(f"Pooled marginal density, {len(cols)} features{tail}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = a.out or os.path.join(
        a.runs[0] if not a.runs[0].endswith(".npy") else os.path.dirname(a.runs[0]),
        "probability_density.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
