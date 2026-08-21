"""Step 2.  Price panel -> log returns, with every drop listed by VALOR.

    python 02_returns.py --data data/prices.csv --out data/returns.csv

Filters, each removing series that break a specific downstream assumption:

  non-positive  load_returns takes np.log.  Isolated non-positive cells are
                masked to NaN (a stray zero placeholder costs one observation,
                not the whole instrument); a column with more than
                --max-nonpos-frac of them is dropped outright.
  coverage      a column missing too much of the calendar drags every other
                column down once rows are inner-joined.
  zero fraction stale or coarsely quantised quotes.  Their "empirical body"
                between the EVT thresholds is a handful of knots.
  EVT feasible  evt.SemiParametricMarginal fits a GPD to x[x < u_lo], u_lo the
                q-quantile.  A point mass at the minimum -- accrual NAVs that
                never fall -- makes that set empty and scipy raises "zero-size
                array to reduction operation minimum".  Not optional.

Every dropped VALOR is printed with the number that caused it.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def report(cols, stat, why, unit=""):
    if len(cols) == 0:
        return
    print(f"\ndropped {len(cols)} VALOR -- {why}:")
    for c in cols:
        print(f"  {c:>12}  {stat[c]:.4g}{unit}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="price panel from 01_panel.py")
    p.add_argument("--out", required=True)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--max-nonpos-frac", type=float, default=0.01,
                   help="above this fraction of non-positive cells, drop the column")
    p.add_argument("--min-coverage", type=float, default=0.98)
    p.add_argument("--ffill-limit", type=int, default=2)
    p.add_argument("--max-zero-frac", type=float, default=0.05)
    p.add_argument("--q-tail", type=float, default=0.05,
                   help="must match fit_returns.py --q-tail")
    p.add_argument("--min-exceedances", type=int, default=30)
    p.add_argument("--test-frac", type=float, default=0.2)
    a = p.parse_args()

    px = pd.read_csv(a.data, index_col=0, parse_dates=True).sort_index()
    px.columns = [str(c) for c in px.columns]
    if a.start:
        px = px[px.index >= pd.Timestamp(a.start)]
    if a.end:
        px = px[px.index <= pd.Timestamp(a.end)]
    print(f"{a.data}: {px.shape[0]} dates x {px.shape[1]} VALOR  "
          f"{px.index[0].date()} -> {px.index[-1].date()}")

    # ---- non-positive prices
    npf = (px <= 0).sum() / px.notna().sum()
    bad = list(px.columns[npf > a.max_nonpos_frac])
    report(bad, npf, f"non-positive price fraction > {a.max_nonpos_frac}")
    px = px.drop(columns=bad)
    iso = int((px <= 0).sum().sum())
    if iso:
        print(f"\nmasked {iso} isolated non-positive cells to NaN across "
              f"{int(((px <= 0).sum() > 0).sum())} VALOR")
        px = px.mask(px <= 0)

    # ---- coverage
    cov = px.notna().mean()
    bad = list(cov.index[cov < a.min_coverage])
    report(bad, cov, f"coverage < {a.min_coverage}")
    px = px.drop(columns=bad)
    if px.shape[1] == 0:
        raise SystemExit("--min-coverage dropped every column")

    before = len(px)
    if a.ffill_limit > 0:
        px = px.ffill(limit=a.ffill_limit)
    px = px.dropna(how="any")
    print(f"\nrows {before} -> {len(px)} after ffill(limit={a.ffill_limit}) + dropna")

    ret = np.log(px / px.shift(1)).dropna(how="any")
    print(f"returns: {ret.shape[0]} x {ret.shape[1]}")

    # ---- stale quotes
    zf = (ret == 0.0).mean()
    bad = list(ret.columns[zf > a.max_zero_frac])
    report(bad, zf, f"zero-return fraction > {a.max_zero_frac}")
    ret = ret.drop(columns=bad)

    # ---- EVT feasibility, on the training rows the marginals are fitted to
    tr = ret.iloc[:int((1 - a.test_frac) * len(ret))]
    n_lo = (tr < tr.quantile(a.q_tail)).sum()
    n_hi = (tr > tr.quantile(1 - a.q_tail)).sum()
    worst = pd.concat([n_lo, n_hi], axis=1).min(axis=1)
    bad = list(ret.columns[worst < a.min_exceedances])
    report(bad, worst, f"fewer than {a.min_exceedances} exceedances in a tail",
           " obs")
    ret = ret.drop(columns=bad)
    if ret.shape[1] == 0:
        raise SystemExit("everything filtered out; loosen the thresholds")

    ret.index.name = "Date"
    ret.to_csv(a.out, float_format="%.8e")
    T, f = ret.shape
    ntr = int((1 - a.test_frac) * T)
    print(f"\nwrote {a.out}: T={T}  f={f}  "
          f"{ret.index[0].date()} -> {ret.index[-1].date()}")
    print(f"train rows {ntr}  exceedances/tail {int(a.q_tail * ntr)}  "
          f"n*f at n=24 is {24 * f} vs {ntr // 24} non-overlapping windows")


if __name__ == "__main__":
    main()
