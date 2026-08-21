"""Step 1.  Raw extract -> one daily price series per VALOR, wide.

Input (semicolon-separated, everything in column A when Excel opens it):
    VALOR;FI_ID;PRICE_TYPE;PRICE_DATE;PRICE;CURRENCY

Output: Date,<valor>,<valor>,...   one column per VALOR, prices.

Three things happen here:

  one quote per VALOR   a VALOR can appear under several PRICE_TYPEs and several
                        CURRENCYs.  Exactly one (PRICE_TYPE, CURRENCY) is kept:
                        whatever --price-type / --currency allow, then the
                        combination with the most observations, ties broken
                        lexically so the choice is reproducible.
  weekends removed      a single VALOR reporting on a Saturday creates a row
                        that is empty for every other VALOR, which then counts
                        against every column's coverage.
  non-daily removed     some VALORs report quarterly.  The median gap between
                        consecutive observations separates the populations by
                        two orders of magnitude (~1-3 days vs ~91), so the
                        threshold is not delicate.  Each one is listed.

Column names are the bare VALOR.  Safe because a Date column is always written:
load_returns drops the header and any column whose first entry is not a float,
and feature_names_from_csv drops the "date" name.  Without a Date column a
numeric header would be parsed as data row 0.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

COLS = ["VALOR", "FI_ID", "PRICE_TYPE", "PRICE_DATE", "PRICE", "CURRENCY"]


def sniff_sep(path: str, encoding: str) -> str:
    with open(path, encoding=encoding, errors="replace") as fh:
        head = fh.readline()
    return max([";", "\t", "|", ","], key=head.count)


def to_float(s: pd.Series) -> pd.Series:
    """1'234.56 / 1 234,56 / 1.234,56 -> float."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = (s.astype(str).str.strip()
         .str.replace("'", "", regex=False)
         .str.replace("\u00a0", "", regex=False)
         .str.replace(" ", "", regex=False))
    comma_dec = t.str.contains(",") & ~t.str.contains(r",\d*\.")
    t = t.mask(comma_dec, t.str.replace(".", "", regex=False)
                          .str.replace(",", ".", regex=False))
    return pd.to_numeric(t.str.replace(",", "", regex=False), errors="coerce")


def to_date(s: pd.Series) -> pd.Series:
    txt = s.astype(str).str.strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y%m%d"):
        out = pd.to_datetime(txt, format=fmt, errors="coerce")
        if out.notna().mean() > 0.95:
            return out.dt.normalize()
    return pd.to_datetime(txt, dayfirst=True, errors="coerce").dt.normalize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--price-type", default=None, help="keep only these, comma-separated")
    p.add_argument("--currency", default=None, help="keep only this currency")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--keep-weekends", action="store_true")
    p.add_argument("--max-median-gap", type=float, default=5.0,
                   help="days; a VALOR whose median gap exceeds this is not daily")
    a = p.parse_args()

    sep = sniff_sep(a.raw, a.encoding)
    df = pd.read_csv(a.raw, sep=sep, dtype=str, encoding=a.encoding,
                     engine="python", skipinitialspace=True)
    df.columns = [re.sub(r"[^A-Z_]", "", c.strip().upper()) for c in df.columns]
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"missing {missing}; found {list(df.columns)} (sep={sep!r})")

    df = df[COLS].copy()
    for c in ("VALOR", "FI_ID", "PRICE_TYPE", "CURRENCY"):
        df[c] = df[c].astype(str).str.strip().str.upper()
    df["PRICE_DATE"] = to_date(df["PRICE_DATE"])
    df["PRICE"] = to_float(df["PRICE"])
    n0 = len(df)
    df = df.dropna(subset=["VALOR", "PRICE_DATE", "PRICE"])
    if a.start:
        df = df[df.PRICE_DATE >= pd.Timestamp(a.start)]
    if a.end:
        df = df[df.PRICE_DATE <= pd.Timestamp(a.end)]
    print(f"raw: {n0:,} rows (sep={sep!r}) -> {len(df):,} parsed, "
          f"{df.VALOR.nunique()} VALOR, "
          f"{df.PRICE_DATE.min().date()} -> {df.PRICE_DATE.max().date()}")
    print(f"  PRICE_TYPE: {sorted(df.PRICE_TYPE.unique())}")
    print(f"  CURRENCY  : {sorted(df.CURRENCY.unique())}")
    per = df.groupby("VALOR").agg(n_pt=("PRICE_TYPE", "nunique"),
                                  n_ccy=("CURRENCY", "nunique"))
    print(f"  VALOR with >1 PRICE_TYPE: {(per.n_pt > 1).sum()}   "
          f">1 CURRENCY: {(per.n_ccy > 1).sum()}")

    # ---- 1. one (PRICE_TYPE, CURRENCY) per VALOR
    if a.price_type:
        df = df[df.PRICE_TYPE.isin([x.strip().upper()
                                    for x in a.price_type.split(",")])]
    if a.currency:
        ccy = a.currency.strip().upper()
        lost = sorted(set(df.VALOR) - set(df.loc[df.CURRENCY == ccy, "VALOR"]))
        df = df[df.CURRENCY == ccy]
        if lost:
            print(f"  --currency {ccy}: dropped {len(lost)} VALOR not quoted "
                  f"in it: {lost}")
    if df.empty:
        raise SystemExit("no rows left after --price-type/--currency")

    cnt = (df.groupby(["VALOR", "PRICE_TYPE", "CURRENCY"], as_index=False)
             .agg(n=("PRICE", "size"))
             .sort_values(["VALOR", "n", "PRICE_TYPE", "CURRENCY"],
                          ascending=[True, False, True, True]))
    best = cnt.drop_duplicates("VALOR")
    df = df.merge(best[["VALOR", "PRICE_TYPE", "CURRENCY"]],
                  on=["VALOR", "PRICE_TYPE", "CURRENCY"], how="inner")
    df = df.drop_duplicates(["VALOR", "PRICE_DATE"], keep="last")
    mix = best.CURRENCY.value_counts().to_dict()
    print(f"\nselected one quote per VALOR; currency mix {mix}")
    if len(mix) > 1:
        print("  WARNING: mixed currencies -- columns are in different units and "
              "their log returns differ by an FX return.  Use --currency.")

    # ---- 2. weekends
    if not a.keep_weekends:
        wk = df.PRICE_DATE.dt.dayofweek >= 5
        if wk.any():
            ndates = df.loc[wk, "PRICE_DATE"].nunique()
            print(f"dropped {int(wk.sum())} weekend rows on {ndates} dates")
            df = df[~wk]

    # ---- 3. non-daily VALORs
    rows = []
    for v, g in df.groupby("VALOR", sort=True):
        d = np.sort(g.PRICE_DATE.unique())
        gap = (np.median(np.diff(d).astype("timedelta64[D]").astype(float))
               if d.size > 1 else np.inf)
        rows.append((v, d.size, gap))
    fr = pd.DataFrame(rows, columns=["VALOR", "n_obs", "median_gap_days"])
    bad = fr[fr.median_gap_days > a.max_median_gap].sort_values("median_gap_days")
    if len(bad):
        print(f"\ndropped {len(bad)} non-daily VALOR "
              f"(median gap > {a.max_median_gap} days):")
        for _, r in bad.iterrows():
            print(f"  {r.VALOR:>12}  {int(r.n_obs):5d} obs   "
                  f"median gap {r.median_gap_days:6.1f} d")
        df = df[~df.VALOR.isin(bad.VALOR)]

    # ---- 4. wide panel
    px = df.pivot(index="PRICE_DATE", columns="VALOR", values="PRICE").sort_index()
    px.index.name = "Date"
    px.columns = [str(c) for c in px.columns]
    px.to_csv(a.out, float_format="%.8f")
    best[best.VALOR.isin(px.columns)].rename(columns={"n": "n_obs"}) \
        .to_csv(a.out.replace(".csv", "_selected.csv"), index=False)

    obs = px.notna().sum()
    print(f"\npanel: {px.shape[0]} dates x {px.shape[1]} VALOR  "
          f"{px.index[0].date()} -> {px.index[-1].date()}")
    print(f"observations per VALOR: min {obs.min()}  median {int(obs.median())}  "
          f"max {obs.max()}   fully-observed dates "
          f"{int(px.notna().all(axis=1).sum())}")
    print(f"wrote {a.out} (+ _selected.csv)")


if __name__ == "__main__":
    main()
