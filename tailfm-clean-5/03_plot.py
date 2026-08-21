"""Step 3.  Plot the series -- prices or returns, all VALORs or a chosen few.

    python 03_plot.py --data data/returns.csv --out fig/returns.png
    python 03_plot.py --data data/prices.csv  --out fig/prices.png
    python 03_plot.py --data data/prices.csv  --out fig/two.png --valors V4156860,V4156861
    python 03_plot.py --data data/prices.csv  --out fig/r.png --to-returns

One panel per VALOR on a shared date axis.  With many columns the panels are
capped by --max-panels and split across pages (fig/x.png, fig/x_02.png, ...):
matplotlib refuses images past 2**16 pixels in either direction, and a single
strip of 300 panels is neither renderable nor readable.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--valors", default=None,
                   help="comma-separated column names; default = all")
    p.add_argument("--to-returns", action="store_true",
                   help="input is prices; plot log returns instead")
    p.add_argument("--max-panels", type=int, default=24)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--overlay", action="store_true",
                   help="all series in one panel instead of a grid")
    a = p.parse_args()

    df = pd.read_csv(a.data, index_col=0, parse_dates=True).sort_index()
    if a.valors:
        want = [v.strip() for v in a.valors.split(",")]
        missing = [v for v in want if v not in df.columns]
        if missing:
            raise SystemExit(f"not in {a.data}: {missing}\n"
                             f"available: {list(df.columns[:10])} ...")
        df = df[want]
    if a.to_returns:
        df = np.log(df / df.shift(1)).dropna(how="all")
    label = "log return" if a.to_returns or "return" in a.data else "level"
    print(f"{a.data}: {df.shape[0]} dates x {df.shape[1]} columns  ({label})")

    if a.overlay:
        fig, ax = plt.subplots(figsize=(12, 6))
        for c in df.columns:
            ax.plot(df.index, df[c], lw=0.7, label=c)
        ax.set_ylabel(label)
        if df.shape[1] <= 12:
            ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(a.out, dpi=130)
        plt.close(fig)
        print(f"wrote {a.out}")
        return

    cols = list(df.columns[:a.max_panels]) if a.max_panels else list(df.columns)
    pages = [cols[i:i + a.max_panels] for i in range(0, len(df.columns), a.max_panels)] \
        if a.max_panels else [list(df.columns)]
    pages = [list(df.columns)[i:i + (a.max_panels or len(df.columns))]
             for i in range(0, df.shape[1], a.max_panels or df.shape[1])]
    stem, ext = os.path.splitext(a.out)

    for k, page in enumerate(pages):
        ncol = min(a.cols, len(page))
        nrow = math.ceil(len(page) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.6 * nrow),
                                 squeeze=False, sharex=True)
        for ax, c in zip(axes.ravel(), page):
            ax.plot(df.index, df[c], lw=0.7)
            ax.set_title(c, fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(page):]:
            ax.axis("off")
        fig.supylabel(label)
        tag = "" if len(pages) == 1 else f" [page {k + 1}/{len(pages)}]"
        fig.suptitle(f"{os.path.basename(a.data)}{tag}")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = a.out if len(pages) == 1 else f"{stem}_{k + 1:02d}{ext}"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"wrote {path}  ({len(page)} panels)")


if __name__ == "__main__":
    main()
