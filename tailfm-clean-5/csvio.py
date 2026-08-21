"""CSV loader shared by fit_returns and the diagnostics."""

from __future__ import annotations

import numpy as np


def _is_float(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def load_returns(path: str, prices: bool) -> np.ndarray:
    """(T, f) log returns.  Drops a header row and any non-numeric Date column."""
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        try:
            arr = np.loadtxt(path, delimiter=",")
        except ValueError:                       # header row and/or Date column
            with open(path) as fh:
                rows = [ln.strip().split(",") for ln in fh if ln.strip()]
            if not all(_is_float(t) for t in rows[0]):
                rows = rows[1:]
            # A column is numeric if every row is a float or empty.  Deciding from
            # rows[0] alone deletes a column that merely happens to be missing on the
            # first date, and empty cells in later rows then raise float('').
            keep = [j for j in range(len(rows[0]))
                    if all(_is_float(r[j]) or not r[j].strip() for r in rows)]
            arr = np.array([[float(r[j]) if r[j].strip() else np.nan
                             for j in keep] for r in rows])
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if prices:
        arr = np.diff(np.log(arr), axis=0)
    if not np.isfinite(arr).all():
        n_full = int(np.isfinite(arr).all(axis=1).sum())
        raise SystemExit(
            f"{path}: {int((~np.isfinite(arr)).sum())} missing/non-finite cells; only "
            f"{n_full} of {arr.shape[0]} rows fully observed.  Run 02_returns.py "
            "(coverage filter + ffill + dropna) first.")
    return arr


def feature_names_from_csv(path: str, f: int) -> list[str]:
    """Use the CSV header for labels if one is present."""
    if path.endswith(".npy"):
        return [f"feat{j}" for j in range(f)]
    with open(path) as fh:
        first = fh.readline().strip().split(",")
    try:
        [float(v) for v in first if v]                    # no header
        return [f"feat{j}" for j in range(f)]
    except ValueError:
        names = [c for c in first if c.lower() != "date"]
        return names if len(names) == f else [f"feat{j}" for j in range(f)]
