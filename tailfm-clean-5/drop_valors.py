"""Drop a hand-picked list of VALOR columns from a returns or prices CSV.

    python drop_valors.py --data data/returns.csv --out data/returns_clean.csv \
        --valors 4155686,4155690,4157124

    python drop_valors.py --data data/returns.csv --out data/returns_clean.csv \
        --file drop.txt          # one VALOR per line, '#' comments allowed

    python drop_valors.py --data data/returns.csv --out data/returns_keep.csv \
        --valors 4155011,4155487 --keep      # keep these instead of dropping

Names must match the CSV header exactly.  Anything not found is reported and the
script stops, rather than silently dropping nothing -- a typo that leaves the
file unchanged is worse than an error, because the run afterwards looks fine.
"""

from __future__ import annotations

import argparse

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--valors", default=None, help="comma-separated column names")
    p.add_argument("--file", default=None, help="text file, one name per line")
    p.add_argument("--keep", action="store_true",
                   help="keep the listed columns instead of dropping them")
    a = p.parse_args()

    names = []
    if a.valors:
        names += [s.strip() for s in a.valors.split(",") if s.strip()]
    if a.file:
        with open(a.file) as fh:
            names += [ln.split("#")[0].strip() for ln in fh
                      if ln.split("#")[0].strip()]
    if not names:
        raise SystemExit("give --valors and/or --file")

    df = pd.read_csv(a.data, index_col=0, parse_dates=True)
    df.columns = [str(c) for c in df.columns]
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise SystemExit(f"not in {a.data}: {missing}\n"
                         f"available (first 10): {list(df.columns[:10])}")

    out = df[[c for c in df.columns if c in set(names)]] if a.keep \
        else df.drop(columns=names)
    out.to_csv(a.out, float_format="%.8e")
    verb = "kept" if a.keep else "dropped"
    print(f"{verb} {len(names)}: {names}")
    print(f"{a.data} {df.shape[1]} cols -> {a.out} {out.shape[1]} cols "
          f"({out.shape[0]} rows)")


if __name__ == "__main__":
    main()
