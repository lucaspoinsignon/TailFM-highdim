"""Step 4.  What dependence structure is actually in the data, before fitting.

    python 04_analyse.py --data data/returns.csv --out fig/dependence.png

Prints, and plots four panels:

  1. rank correlation heatmap (Spearman -- the copula-level quantity, unaffected
     by the heterogeneous scales across VALOR)
  2. eigenvalue spectrum of the pseudo-observations, log scale, with the
     effective rank at several thresholds.  This is the single most useful
     number here: f columns with effective rank k means the flow is asked to
     model an f-dimensional law supported near a k-dimensional set.  Directions
     below the floor carry no variance, so the isotropically weighted CFM loss
     gives them no gradient -- yet lambda depends entirely on them.
  3. distribution of lambda_hat(q0) over all pairs, against the q0 line that
     independence would give
  4. the most nearly comonotone pairs, which no absolutely-continuous
     generative model can reproduce

The rank transform is applied directly, so this needs no trained model and no
marginals.pkl -- ranks are invariant under the EVT PIT, which is monotone per
coordinate, so the spectrum here is the same one the model sees in z-space.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pseudo_obs(x: np.ndarray) -> np.ndarray:
    n, f = x.shape
    u = np.empty((n, f))
    for j in range(f):
        u[:, j] = (np.argsort(np.argsort(x[:, j])) + 1.0) / (n + 1.0)
    return u


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="returns CSV from 02_returns.py")
    p.add_argument("--out", default="fig/dependence.png")
    p.add_argument("--q0", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.2,
                   help="analyse the training rows only")
    p.add_argument("--top", type=int, default=15)
    a = p.parse_args()

    df = pd.read_csv(a.data, index_col=0, parse_dates=True).sort_index()
    tr = df.iloc[:int((1 - a.test_frac) * len(df))]
    x = tr.to_numpy(dtype=float)
    T, f = x.shape
    names = list(df.columns)
    print(f"{a.data}: analysing {T} training rows x {f} VALOR\n")

    u = pseudo_obs(x)
    uc = (u - u.mean(0)) / u.std(0)
    rho = (uc.T @ uc) / T
    b = (u < a.q0).astype(float)
    lam = (b.T @ b) / (a.q0 * T)
    iu = np.triu_indices(f, 1)

    s = np.linalg.svd(u - u.mean(0), compute_uv=False) ** 2
    s = s / s.max()
    print("effective rank of the rank-transformed panel:")
    for t in (1e-12, 1e-8, 1e-6, 1e-4, 1e-2):
        print(f"  eigenvalues below {t:.0e} of max: {int((s < t).sum()):4d}"
              f"   -> effective rank {int((s >= t).sum())}")
    cum = np.cumsum(s) / s.sum()
    for v in (0.90, 0.99, 0.999):
        print(f"  components for {v:.1%} of variance: "
              f"{int(np.searchsorted(cum, v)) + 1} of {f}")

    print(f"\nSpearman |rho| over {iu[0].size} pairs: "
          f"mean {np.abs(rho[iu]).mean():.3f}  q95 {np.quantile(np.abs(rho[iu]), .95):.3f}"
          f"  max {np.abs(rho[iu]).max():.4f}")
    print(f"lambda_hat({a.q0}): mean {lam[iu].mean():.3f}  "
          f"q95 {np.quantile(lam[iu], .95):.3f}  max {lam[iu].max():.3f}   "
          f"(independence gives {a.q0:.3f})")
    print(f"pairs with lambda > 0.8: {int((lam[iu] > 0.8).sum())}   "
          f"|rho_S| > 0.99: {int((np.abs(rho[iu]) > 0.99).sum())}")

    order = np.argsort(-lam[iu])[:a.top]
    print(f"\nmost tail-dependent pairs:")
    for k in order:
        i, j = iu[0][k], iu[1][k]
        print(f"  lambda {lam[i, j]:.3f}  rho_S {rho[i, j]:+.4f}   "
              f"{names[i]} / {names[j]}")

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    im = ax[0][0].imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0][0].set_title(f"Spearman rank correlation ({f} VALOR)")
    fig.colorbar(im, ax=ax[0][0], fraction=0.046)

    ax[0][1].semilogy(np.arange(1, f + 1), np.maximum(s, 1e-40), ".-", ms=3)
    for t in (1e-4, 1e-8):
        ax[0][1].axhline(t, ls="--", lw=0.8, color="grey")
        ax[0][1].text(f * 0.55, t * 1.5, f"rank@{t:.0e} = {int((s >= t).sum())}",
                      fontsize=8, color="grey")
    ax[0][1].set_title("eigenvalue spectrum of pseudo-observations")
    ax[0][1].set_xlabel("component"); ax[0][1].set_ylabel("eigenvalue / max")

    ax[1][0].hist(lam[iu], bins=80, color="C0")
    ax[1][0].axvline(a.q0, color="k", ls="--", lw=1,
                     label=f"independence = {a.q0}")
    ax[1][0].set_yscale("log")
    ax[1][0].set_title(rf"$\hat\lambda({a.q0})$ over all pairs")
    ax[1][0].set_xlabel(r"$\hat\lambda$"); ax[1][0].legend(fontsize=8)

    ax[1][1].plot(np.abs(rho[iu]), lam[iu], ".", ms=1, alpha=0.3)
    ax[1][1].set_xlabel(r"$|\rho_S|$"); ax[1][1].set_ylabel(r"$\hat\lambda$")
    ax[1][1].set_title("tail dependence vs rank correlation")
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
