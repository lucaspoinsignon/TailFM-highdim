"""One table across run directories: clustering, tail dependence, portfolio risk.

    python summarize_runs.py --data data/returns_clean.csv runs/final runs/other

Everything is scored against the TRAIN windows -- the target distribution for a
scenario generator.  Column references, which are not the obvious ones:

  sqACF1/2   mean over features of the within-window squared-return ACF minus the iid
             null -1/(n-1).  0 means the generated windows are exchangeable in time;
             the target is the `real` row.

  lam        mean lambda_hat(q0) over all pairs, with corr / slope against the real
             pairwise values.  A POPULATION-correct generator scores about corr 0.91 /
             slope 0.86 at n = 1236, not 1.0, because lambda_hat(real) is itself
             measured with error (regression dilution).

  err/flr    mean|lam_gen - lam_real| over mean|lam_halfA - lam_halfB|.  The null
             splits the real rows into halves of n/2, so a perfect model scores ~0.5,
             not 1.0.  A worse-fitting model also scores higher, so this does not
             separate fit quality from memorisation -- use novelty.py for that.

  VaR/CVaR   h-step equal-weight portfolio loss at alpha = 0.995 from the generated
             windows.  Comparable across runs only; the real sample has ~50
             non-overlapping h-step blocks, so its own quantile here is not estimable.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from csvio import load_returns
from tailfm import make_windows, portfolio_losses
from tailfm.evaluate import acf, pseudo_obs, _pool


def sq_acf_excess(W: np.ndarray, n: int, max_lag: int = 3) -> np.ndarray:
    """Mean-over-features within-window squared-return ACF, minus the iid null."""
    return np.array([acf(W[:, :, j] ** 2, max_lag)
                     for j in range(W.shape[2])]).mean(0) + 1.0 / (n - 1)


def lam_matrix(u: np.ndarray, q0: float) -> np.ndarray:
    b = (u < q0).astype(np.float32)
    return (b.T @ b) / (q0 * b.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--q0", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()

    r = load_returns(a.data, a.prices)
    T, f = r.shape
    train = r[:int((1.0 - a.test_frac) * T)]
    real = make_windows(train, a.n, 1)
    w = np.full(f, 1.0 / f)
    iu = np.triu_indices(f, 1)

    Lr = lam_matrix(pseudo_obs(_pool(real, a.max_rows, a.seed)), a.q0)[iu]
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(train.shape[0]); h = train.shape[0] // 2
    La = lam_matrix(pseudo_obs(train[idx[:h]]), a.q0)[iu]
    Lb = lam_matrix(pseudo_obs(train[idx[h:2 * h]]), a.q0)[iu]
    floor = np.abs(La - Lb).mean()

    ar = sq_acf_excess(real, a.n)
    sh = real.copy()
    for i in range(sh.shape[0]):
        sh[i] = sh[i][rng.permutation(a.n)]
    ash = sq_acf_excess(sh, a.n)

    print(f"{a.data}: train {train.shape}, windows {real.shape}, q0={a.q0}, "
          f"h={a.horizon}\n")
    hdr = (f"{'run':>16}{'sqACF1':>9}{'sqACF2':>9}{'lam':>8}{'corr':>7}"
           f"{'slope':>7}{'err/flr':>9}{'VaR.995':>10}{'CVaR.995':>10}")
    print(hdr); print("-" * len(hdr))
    print(f"{'real (target)':>16}{ar[0]:+9.4f}{ar[1]:+9.4f}{Lr.mean():8.3f}"
          f"{1.0:7.3f}{1.0:7.3f}{'-':>9}{'-':>10}{'-':>10}")
    print(f"{'shuffled (null)':>16}{ash[0]:+9.4f}{ash[1]:+9.4f}"
          f"{'-':>8}{'-':>7}{'-':>7}{'-':>9}{'-':>10}{'-':>10}")
    print("-" * len(hdr))

    for run in a.runs:
        p = os.path.join(run, "generated_windows.npy")
        if not os.path.exists(p):
            print(f"{os.path.basename(run):>16}  (no generated_windows.npy)")
            continue
        gen = np.load(p)
        ag = sq_acf_excess(gen, a.n)
        Lg = lam_matrix(pseudo_obs(_pool(gen, a.max_rows, a.seed)), a.q0)[iu]
        L = portfolio_losses(gen, weights=w, horizon=a.horizon)
        print(f"{os.path.basename(run):>16}{ag[0]:+9.4f}{ag[1]:+9.4f}"
              f"{Lg.mean():8.3f}{np.corrcoef(Lr, Lg)[0, 1]:7.3f}"
              f"{np.polyfit(Lr, Lg, 1)[0]:7.3f}"
              f"{np.abs(Lg - Lr).mean() / floor:9.2f}"
              f"{np.quantile(L, 0.995):10.5f}{L[L >= np.quantile(L, 0.995)].mean():10.5f}")
        del gen

    print(f"\nreferences: sqACF -> the 'real' row (0.0 = time-exchangeable);  "
          f"err/flr -> ~0.5 for a population-correct\n"
          f"            model, but a worse fit also scores higher -- use novelty.py "
          f"to test memorisation;\n"
          f"            corr/slope -> ~0.91/0.86 for a population-correct model at "
          f"n={train.shape[0]}, not 1.0.")


if __name__ == "__main__":
    main()
