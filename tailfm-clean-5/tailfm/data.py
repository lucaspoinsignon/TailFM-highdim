"""Windowing utility."""

from __future__ import annotations

import numpy as np


def make_windows(series: np.ndarray, n: int, stride: int = 1) -> np.ndarray:
    """(T, f) series -> (N, n, f) overlapping windows."""
    series = np.asarray(series, dtype=float)
    starts = np.arange(0, series.shape[0] - n + 1, stride)
    return np.stack([series[s:s + n] for s in starts], axis=0)
