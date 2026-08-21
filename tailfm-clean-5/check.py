"""Validate a CSV before handing it to fit_returns.py.

    python check.py --data data/prices.csv --prices     # after step 1
    python check.py --data data/returns_clean.csv       # after step 2

Exits non-zero if anything would break the run, so it can gate a script.  Checks, in
the order they would bite:

  header      a bare numeric header (VALOR ids with no prefix) parses as floats, so
              np.loadtxt reads it as row 0 and every id becomes a return of ~1e6 with
              no error raised;
  positivity  load_returns takes np.log, so a non-positive price yields -inf or nan;
  finiteness  after differencing;
  EVT         the GPD fits are actually run, on the same training rows the model will
              use.  A point mass at the minimum makes the exceedance set empty and
              scipy raises; and a fit can succeed and still transform an observation
              to |z| ~ 1e4, which is what destroys training, and only fitting reveals
              it.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from csvio import load_returns, feature_names_from_csv

FAIL: list[str] = []
WARN: list[str] = []


def fail(msg: str) -> None:
    FAIL.append(msg); print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    WARN.append(msg); print(f"  WARN  {msg}")


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--prices", action="store_true",
                   help="--data holds prices, not returns (same flag as fit_returns)")
    p.add_argument("--q-tail", type=float, default=0.05)
    p.add_argument("--nu", type=float, default=5.0)
    p.add_argument("--shrink-c", type=float, default=1.0)
    p.add_argument("--no-shrink", action="store_true")
    p.add_argument("--test-frac", type=float, default=0.2)
    a = p.parse_args()
    print(f"checking {a.data}  (--prices={a.prices}, q_tail={a.q_tail}, nu={a.nu}, "
          f"shrink={'off' if a.no_shrink else f'c={a.shrink_c}'})\n")

    # ---- header -------------------------------------------------------------
    # Load the raw columns first (prices=False never takes a log), so a non-positive
    # price is reported by name instead of surfacing from inside np.log.
    print("header")
    try:
        lv = load_returns(a.data, False)
    except SystemExit as e:
        fail(str(e)); sys.exit(1)
    f = lv.shape[1]
    names = feature_names_from_csv(a.data, f)
    if names and names[0].startswith("feat") and not a.data.endswith(".npy"):
        fail("no usable header: names came back as feat0, feat1, ... .  If the header "
             "is bare VALOR numbers it was parsed as DATA row 0 -- every id became a "
             "return of ~1e6.  Prefix the names (V4156860) and rebuild.")
    else:
        ok(f"{f} named columns, e.g. {names[:3]}")
        if all(nm.replace(".", "").replace("-", "").isdigit() for nm in names):
            warn("column names are purely numeric.  They parse here only because a "
                 "non-numeric Date column is present; without it np.loadtxt would "
                 "read the header as data.  Prefix them (V4156860).")

    # ---- levels -------------------------------------------------------------
    if a.prices:
        print("\nprices")
        nonpos = np.flatnonzero((lv <= 0).any(axis=0))
        if nonpos.size:
            fail(f"{nonpos.size} columns contain a non-positive price, so log is "
                 f"undefined: {[names[j] for j in nonpos[:6]]}")
            print(f"\n{'-' * 60}\nFAILED: fix or drop those columns, then re-run.")
            sys.exit(1)
        ok(f"all {f} columns strictly positive (min {lv.min():.6g})")
        if np.abs(lv).mean() < 0.5:
            warn(f"mean |level| is {np.abs(lv).mean():.3g} -- this looks like returns, "
                 "not prices.  Drop --prices?")

    r = load_returns(a.data, a.prices)
    T, f = r.shape
    names = feature_names_from_csv(a.data, f)
    ok(f"loads as (T={T}, f={f})")

    # ---- returns -----------------------------------------------------------
    print("\nreturns")
    if not np.isfinite(r).all():
        fail(f"{int((~np.isfinite(r)).sum())} non-finite returns")
    else:
        ok(f"all finite, sd {r.std():.4g}, range [{r.min():.4g}, {r.max():.4g}]")
    if np.abs(r).mean() > 1.0:
        fail(f"mean |return| is {np.abs(r).mean():.3g}.  Log returns are ~1e-3; this "
             "file is probably price levels -- add --prices.")

    tr = r[:int((1.0 - a.test_frac) * T)]
    ok(f"training rows {tr.shape[0]}  ->  {int(a.q_tail * tr.shape[0])} exceedances "
       f"per tail per column")

    # ---- EVT ----------------------------------------------------------------
    print("\nEVT: fitting marginals on the training rows")
    from tailfm.evt import MarginalEnsemble
    from evt_shrink import shrink_ensemble
    try:
        marg = MarginalEnsemble(q_tail=a.q_tail, nu=a.nu).fit(tr)
    except ValueError as e:
        fail(str(e))
        print(f"\n{'-' * 60}\nFAILED: fit_returns.py will raise on this file.")
        sys.exit(1)
    ok(f"{2 * f} GPD fits succeeded  (nu = {marg.nu_:.3f})")
    if not a.no_shrink:
        print("    pooling xi across features (empirical Bayes):")
        shrink_ensemble(marg, tr, c=a.shrink_c)
    su = marg.summary()
    print(f"    exceedances: lower {su['n_exc_lo'].min()}-{su['n_exc_lo'].max()}, "
          f"upper {su['n_exc_hi'].min()}-{su['n_exc_hi'].max()}")
    for side, xi in (("lower", su["xi_lo"]), ("upper", su["xi_hi"])):
        print(f"    xi_{side}: median {np.median(xi):+.3f}  "
              f"[{xi.min():+.3f}, {xi.max():+.3f}]  "
              f"| xi>1: {int((xi > 1).sum())}  xi<0: {int((xi < 0).sum())}")

    # ---- the operational check ----------------------------------------------
    z = marg.transform(tr)
    mz = np.abs(z).max(axis=0)
    if not np.isfinite(z).all():
        fail(f"{int((~np.isfinite(z)).sum())} non-finite values after the PIT")
    bad = np.flatnonzero(mz > 100.0)
    if bad.size:
        fail(f"{bad.size} features transform to |z| > 100.  A correct fit keeps max|z| "
             f"under ~100 at any tail index, so these marginals are mis-fitted and "
             f"will dominate the training gradient: "
             + ", ".join(f"{names[j]}({mz[j]:.0f})" for j in bad[:6]))
    else:
        ok(f"max|z| = {mz.max():.1f} over all features (correct fits stay under ~100)")
    warnz = np.flatnonzero((mz > 50.0) & (mz <= 100.0))
    if warnz.size:
        warn(f"{warnz.size} features with 50 < max|z| <= 100: plausible but on the "
             "edge of what a correct fit produces")

    print(f"\n{'-' * 60}")
    if FAIL:
        print(f"FAILED: {len(FAIL)} blocking issue(s), {len(WARN)} warning(s).")
        print("fit_returns.py will crash or produce invalid output on this file.")
        sys.exit(1)
    print(f"PASSED with {len(WARN)} warning(s).  Ready for fit_returns.py.")


if __name__ == "__main__":
    main()
