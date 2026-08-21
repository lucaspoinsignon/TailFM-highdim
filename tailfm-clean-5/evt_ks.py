"""PIT goodness-of-fit per feature: is the fitted marginal actually right?

    python evt_ks.py --data data/returns_clean.csv --out fig/ks.png

If F_j were the true CDF then F_j(X_j) ~ U(0,1), hence z = T_nu^{-1}(F_j(X_j)) ~ t_nu.
The flow is trained on z under a squared-error loss, so a mis-specified marginal does
not stay confined to that marginal -- one observation at |z| = 1e4 contributes 1e8
times a typical one and dominates the gradient for its batch.

Reported per feature, on train and on the held-out rows:

    KS       sup_u |F_hat_n(u) - u| on the PIT values, with its p-value
    AD       Anderson-Darling style weighted statistic, which unlike KS puts most of
             its weight in the tails
    max|z|   the largest transformed value; under t_nu with n ~ 1200 the expected
             maximum is ~8-15, and anything past ~50 means F_j returned a probability
             the data contradicts
    n_beyond test observations outside the training range, where the GPD extrapolates

Two caveats on the p-values.  The train columns are not a valid test -- the body is the
empirical CDF of those rows, so the in-sample PIT values are the plotting positions and
A^2 is near 0 by construction.  And the test p-value assumes iid PIT values, which
return series are not: on a stochastic-volatility t_5 process a correctly specified
marginal rejects 30-55% of the time at nominal 5%.  Compare the rejection rate against
a placebo split inside the training window rather than against 5%.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from csvio import load_returns, feature_names_from_csv
from tailfm.evt import MarginalEnsemble       # direct: avoids the torch import in
from evt_shrink import shrink_ensemble        # tailfm/__init__


def ad_stat(u: np.ndarray) -> float:
    """Anderson-Darling A^2 for uniformity; weights the tails heavily."""
    u = np.sort(np.clip(u, 1e-12, 1 - 1e-12))
    n = u.size
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2 * i - 1) * (np.log(u) + np.log1p(-u[::-1]))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--prices", action="store_true")
    p.add_argument("--marginals", default=None,
                   help="reuse a fitted marginals.pkl instead of refitting")
    p.add_argument("--q-tail", type=float, default=0.05)
    p.add_argument("--nu", type=float, default=5.0)
    p.add_argument("--shrink-c", type=float, default=1.0)
    p.add_argument("--no-shrink", action="store_true")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--out", default="fig/ks.png")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--csv", default="evt_ks.csv")
    a = p.parse_args()

    r = load_returns(a.data, a.prices)
    T, f = r.shape
    names = feature_names_from_csv(a.data, f)
    split = int((1 - a.test_frac) * T)
    tr, te = r[:split], r[split:]
    print(f"{a.data}: train {tr.shape}  test {te.shape}")

    if a.marginals:
        import pickle
        marg = pickle.load(open(a.marginals, "rb"))
        print(f"loaded {a.marginals}  (nu={marg.nu_:.3f}, q_tail={marg.q_tail})")
    else:
        marg = MarginalEnsemble(q_tail=a.q_tail, nu=a.nu).fit(tr)
        if not a.no_shrink:
            print(f"pooling xi across features (empirical Bayes, c={a.shrink_c}):")
            shrink_ensemble(marg, tr, c=a.shrink_c)
        print(f"fitted marginals on train rows (nu={marg.nu_:.3f}, q_tail={a.q_tail}, "
              f"shrink={'off' if a.no_shrink else f'c={a.shrink_c}'})")

    rows = []
    for j, m in enumerate(marg.marginals_):
        u_tr, u_te = m.cdf(tr[:, j]), m.cdf(te[:, j])
        z_tr, z_te = m.transform(tr[:, j]), m.transform(te[:, j])
        ks_tr, ks_te = stats.kstest(u_tr, "uniform"), stats.kstest(u_te, "uniform")
        rows.append(dict(
            VALOR=names[j], xi_lo=m.xi_lo_, xi_hi=m.xi_hi_,
            KS_train=ks_tr.statistic, p_train=ks_tr.pvalue,
            KS_test=ks_te.statistic, p_test=ks_te.pvalue,
            AD_train=ad_stat(u_tr), AD_test=ad_stat(u_te),
            q_lo=m.q_lo_, q_hi=m.q_hi_,
            n_exc_lo=m.n_exc_lo_, n_exc_hi=m.n_exc_hi_,
            max_abs_z=max(np.abs(z_tr).max(), np.abs(z_te).max()),
            n_beyond=int((te[:, j] < tr[:, j].min()).sum()
                         + (te[:, j] > tr[:, j].max()).sum()),
        ))
    d = pd.DataFrame(rows).sort_values("AD_test", ascending=False)
    d.to_csv(a.csv, index=False)

    with pd.option_context("display.width", 200):
        print(f"\nworst {a.top} features by test Anderson-Darling:")
        print(d.head(a.top).to_string(index=False, float_format="%.4g"))
    print(f"\nrejected at 5% (test KS): {int((d.p_test < 0.05).sum())} of {f}")
    print(f"xi_lo > 0.5: {int((d.xi_lo > 0.5).sum())}   "
          f"max|z| > 50: {int((d.max_abs_z > 50).sum())}   "
          f"max|z| > 500: {int((d.max_abs_z > 500).sum())}")
    print(f"full table -> {a.csv}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0][0].hist(d.p_test.clip(0, 1), bins=25, color="C0")
    ax[0][0].axhline(f / 25, ls="--", color="k", lw=1, label="uniform if all fit")
    ax[0][0].set_title("test KS p-values"); ax[0][0].set_xlabel("p")
    ax[0][0].legend(fontsize=8)

    ax[0][1].scatter(d.xi_lo, d.max_abs_z, s=8, alpha=.6)
    ax[0][1].set_yscale("log"); ax[0][1].axhline(50, ls="--", color="k", lw=1)
    ax[0][1].axvline(0.5, ls="--", color="r", lw=1)
    ax[0][1].set_xlabel(r"$\hat\xi_{lower}$"); ax[0][1].set_ylabel("max |z|")
    ax[0][1].set_title(r"extreme $z$ comes from runaway $\hat\xi$")

    ax[1][0].scatter(d.AD_train, d.AD_test, s=8, alpha=.6)
    lim = [0, float(np.nanpercentile(d.AD_test, 99))]
    ax[1][0].plot(lim, lim, "k--", lw=1); ax[1][0].set_xlim(lim); ax[1][0].set_ylim(lim)
    ax[1][0].set_xlabel("AD train (optimistic)"); ax[1][0].set_ylabel("AD test")
    ax[1][0].set_title("in-sample vs out-of-sample fit")

    worst = int(np.argmax(d.AD_test.values))
    j = names.index(d.VALOR.values[worst])
    u = np.sort(marg.marginals_[j].cdf(te[:, j]))
    ax[1][1].plot(np.linspace(0, 1, u.size), u, lw=1.5,
                  label=f"{d.VALOR.values[worst]} (worst)")
    jb = names.index(d.VALOR.values[-1])
    ub = np.sort(marg.marginals_[jb].cdf(te[:, jb]))
    ax[1][1].plot(np.linspace(0, 1, ub.size), ub, lw=1.5,
                  label=f"{d.VALOR.values[-1]} (best)")
    ax[1][1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1][1].set_xlabel("uniform quantile"); ax[1][1].set_ylabel("PIT quantile")
    ax[1][1].set_title("PP plot of the test PIT values"); ax[1][1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.out, dpi=140); plt.close(fig)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
