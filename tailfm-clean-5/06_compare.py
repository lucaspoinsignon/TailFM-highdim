"""Step 6.  Real vs generated return windows, side by side.

    python 06_compare.py --data data/returns.csv --run runs/main --out fig/windows.png
    python 06_compare.py --data data/returns.csv --run runs/main --out fig/w.png \
        --valors V4156860,V4156861,V4156862,V4156863 --n-windows 2

Left column: windows drawn from the real training set.  Right column: windows
drawn from generated_windows.npy.  The selected VALORs are overlaid in each
panel, so what is being compared is the *joint* path -- whether the series move
together, and whether they move together in the extremes -- not the marginals,
which the EVT PIT matches by construction and which say nothing about the part
of the model that can fail.

Windows are picked at matched loss quantiles rather than at random, so a calm
real window is not being compared against an extreme generated one.  --rank 1.0
shows the worst window in each sample, which is the interesting comparison for a
tail model; the default spreads the picks across the loss distribution.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from csvio import load_returns, feature_names_from_csv


def windows(x: np.ndarray, n: int, stride: int = 1) -> np.ndarray:
    idx = np.arange(0, len(x) - n + 1, stride)
    return np.stack([x[i:i + n] for i in idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True, help="outdir holding generated_windows.npy")
    p.add_argument("--out", default="fig/windows.png")
    p.add_argument("--prices", action="store_true")
    p.add_argument("--valors", default=None,
                   help="comma-separated names; default = first 4 columns")
    p.add_argument("--n-windows", type=int, default=2)
    p.add_argument("--n", type=int, default=24, help="window length, as fitted")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--rank", type=float, default=None,
                   help="loss quantile to select, e.g. 1.0 = worst window")
    p.add_argument("--cumulative", action="store_true",
                   help="plot cumulative return within the window")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    r = load_returns(a.data, a.prices)
    T, f = r.shape
    names = feature_names_from_csv(a.data, f)

    sel = [s.strip() for s in a.valors.split(",")] if a.valors else names[:4]
    miss = [s for s in sel if s not in names]
    if miss:
        raise SystemExit(f"not in {a.data}: {miss}\navailable: {names[:10]} ...")
    cols = [names.index(s) for s in sel]

    real = windows(r[:int((1 - a.test_frac) * T)], a.n)
    gen = np.load(f"{a.run}/generated_windows.npy")
    print(f"real windows {real.shape}   generated {gen.shape}   showing {sel}")

    def pick(w):
        loss = -(w[:, :, cols].mean(axis=2)).sum(axis=1)   # equal-weight loss
        o = np.argsort(loss)
        if a.rank is not None:
            i = min(int(a.rank * (len(o) - 1)), len(o) - 1)
            return o[max(i - a.n_windows + 1, 0):i + 1][::-1]
        qs = np.linspace(0.5, 0.995, a.n_windows)
        return [o[int(q * (len(o) - 1))] for q in qs]

    ir, ig = pick(real), pick(gen)
    step = np.arange(a.n)
    nw = a.n_windows
    fig, axes = plt.subplots(nw, 2, figsize=(12, 3.0 * nw), squeeze=False,
                             sharey=True)
    for row in range(nw):
        for col, (w, idx, tag) in enumerate(
                ((real, ir, "real"), (gen, ig, "generated"))):
            ax = axes[row][col]
            y = w[idx[row]][:, cols]
            if a.cumulative:
                y = np.cumsum(y, axis=0)
            for k, s in enumerate(sel):
                ax.plot(step, y[:, k], marker="o", ms=3, lw=1.2, label=s)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"{tag}  window {idx[row]}", fontsize=10)
            ax.set_xlabel("step within window")
            if col == 0:
                ax.set_ylabel("cumulative return" if a.cumulative else "log return")
            if row == 0 and col == 1:
                ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"Joint {a.n}-step windows, {len(sel)} VALOR "
                 f"({'worst' if a.rank else 'matched loss quantiles'})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.out, dpi=140)
    plt.close(fig)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
