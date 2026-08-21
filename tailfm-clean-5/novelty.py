"""Are the generated windows new, or reshuffled training history?

    python novelty.py --data data/returns_clean.csv --runs runs/final

Every other diagnostic scores a marginal or a pairwise summary, and a model that simply
replayed the training windows would pass all of them perfectly.  For a scenario
generator that is the failure that matters: if the output is the empirical
distribution, resampling the training windows does the same job for free.

Nearest-neighbour distance in z-space (each coordinate standardised by its training sd,
so no single loud feature dominates the metric):

    d_gen   for each generated window, distance to its closest training window
    d_real  for each training window, distance to its closest OTHER training window
    d_boot  the same for a bootstrap resample of the training set, which is 0 by
            construction and is reported as a check that the metric detects copying

Read median(d_gen) / median(d_real): ~1 means generated windows sit as far from the
training set as training windows sit from each other, which is what a sample from the
same law looks like; << 1 means interpolating inside the observed set; ~0 means
copying.  The minimum matters as much as the median -- a single d_gen near 0 is one
training window reproduced outright.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from csvio import load_returns
from tailfm import make_windows


def nn_dist(A: np.ndarray, B: np.ndarray, chunk: int = 512,
            exclude_self: bool = False) -> np.ndarray:
    """For each row of A, the distance to its nearest row of B. A, B are (N, D)."""
    b2 = (B ** 2).sum(1)
    out = np.empty(A.shape[0])
    for s in range(0, A.shape[0], chunk):
        a = A[s:s + chunk]
        d2 = (a ** 2).sum(1)[:, None] + b2[None, :] - 2.0 * (a @ B.T)
        if exclude_self:                      # drop the zero on the diagonal block
            for i in range(a.shape[0]):
                d2[i, s + i] = np.inf
        out[s:s + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--n-gen", type=int, default=3000,
                    help="generated windows to score (the full set is not needed "
                         "and the distance matrix is O(n_gen * n_train * n * f))")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()

    r = load_returns(a.data, a.prices)
    T, f = r.shape
    train = r[:int((1.0 - a.test_frac) * T)]
    real = make_windows(train, a.n, 1)
    sd = train.std(axis=0)                                    # standardise per feature
    flat = lambda W: (W / sd).reshape(W.shape[0], -1)
    R = flat(real)
    rng = np.random.default_rng(a.seed)

    d_real = nn_dist(R, R, exclude_self=True)
    boot = R[rng.integers(0, R.shape[0], min(a.n_gen, R.shape[0]))]
    d_boot = nn_dist(boot, R)
    med = np.median(d_real)

    print(f"{a.data}: {real.shape[0]} training windows of shape "
          f"({a.n}, {f}), z-standardised\n")
    print(f"{'sample':>16}{'median d':>11}{'ratio':>8}{'min d':>10}"
          f"{'q05':>10}{'frac < 0.5x':>13}")
    print("-" * 68)
    print(f"{'real (LOO)':>16}{med:11.2f}{1.00:8.2f}{d_real.min():10.2f}"
          f"{np.quantile(d_real, .05):10.2f}{(d_real < .5*med).mean():13.3f}")
    print(f"{'bootstrap copy':>16}{np.median(d_boot):11.2f}"
          f"{np.median(d_boot)/med:8.2f}{d_boot.min():10.2f}"
          f"{np.quantile(d_boot, .05):10.2f}{(d_boot < .5*med).mean():13.3f}")
    print("-" * 68)

    for run in a.runs:
        p = os.path.join(run, "generated_windows.npy")
        if not os.path.exists(p):
            print(f"{os.path.basename(run):>16}  (no generated_windows.npy)")
            continue
        gen = np.load(p)
        idx = rng.choice(gen.shape[0], min(a.n_gen, gen.shape[0]), replace=False)
        G = flat(gen[np.sort(idx)])
        del gen
        d = nn_dist(G, R)
        print(f"{os.path.basename(run):>16}{np.median(d):11.2f}"
              f"{np.median(d)/med:8.2f}{d.min():10.2f}"
              f"{np.quantile(d, .05):10.2f}{(d < .5*med).mean():13.3f}")

    print("\nratio ~1 = generated windows sit as far from the training set as "
          "training\nwindows sit from each other;  <<1 = interpolating inside it;  "
          "~0 = copying.\n'frac < 0.5x' is the share of generated windows closer to "
          "a training window\nthan half the typical real-to-real distance.")


if __name__ == "__main__":
    main()
