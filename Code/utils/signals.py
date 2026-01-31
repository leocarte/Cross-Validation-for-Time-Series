from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union, Sequence
import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]


def _to_1d_array(x: ArrayLike) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("x must be a 1D array of inputs (e.g., time index or grid).")
    return x_arr


def standardize_signal(y: np.ndarray, eps: float = 1e-12) -> np.ndarray:

    mu = float(np.mean(y))
    sd = float(np.std(y))
    if sd < eps:
        return y - mu
    return (y - mu) / sd




def smooth_trend(
    x: ArrayLike,
    *,
    standardize: bool = True,
    degree: int = 3,
    sinus_amp: float = 0.3,
    sinus_period: Optional[float] = None
) -> np.ndarray:

    x_arr = _to_1d_array(x)

    xmin, xmax = float(np.min(x_arr)), float(np.max(x_arr))
    if np.isclose(xmin, xmax):
        z = np.zeros_like(x_arr)
    else:
        z = 2.0 * (x_arr - xmin) / (xmax - xmin) - 1.0

    y_poly = np.zeros_like(z)
    for p in range(1, degree + 1):
        y_poly += (0.5 ** p) * (z ** p)  

    span = xmax - xmin if not np.isclose(xmin, xmax) else 1.0
    period = sinus_period if sinus_period is not None else max(span, 1.0)
    y_sin = sinus_amp * np.sin(2.0 * np.pi * (x_arr - xmin) / period)

    y = y_poly + y_sin
    return standardize_signal(y) if standardize else y




def nonlinear_curvature(
    x: ArrayLike,
    *,
    standardize: bool = True,
    kink: float = 0.0,
    strength: float = 1.0
) -> np.ndarray:

    x_arr = _to_1d_array(x)

    xmin, xmax = float(np.min(x_arr)), float(np.max(x_arr))
    if np.isclose(xmin, xmax):
        z = np.zeros_like(x_arr)
    else:
        z = 2.0 * (x_arr - xmin) / (xmax - xmin) - 1.0

    s = np.tanh(2.5 * strength * z)

    gate = 1.0 / (1.0 + np.exp(-8.0 * (z - kink)))
    bend_left = -0.6 * (z ** 3)
    bend_right = 0.9 * (z ** 3) + 0.2 * z
    bend = (1 - gate) * bend_left + gate * bend_right

    y = 0.7 * s + 0.6 * bend
    return standardize_signal(y) if standardize else y




def local_bump(
    x: ArrayLike,
    *,
    standardize: bool = True,
    centers: Optional[Sequence[float]] = None,
    widths: Optional[Sequence[float]] = None,
    amplitudes: Optional[Sequence[float]] = None
) -> np.ndarray:

    x_arr = _to_1d_array(x)
    xmin, xmax = float(np.min(x_arr)), float(np.max(x_arr))
    span = xmax - xmin if not np.isclose(xmin, xmax) else 1.0

    if centers is None:
        centers = [xmin + 0.7 * span]
    if widths is None:
        widths = [0.08 * span if span > 0 else 1.0]
    if amplitudes is None:
        amplitudes = [1.0]

    if not (len(centers) == len(widths) == len(amplitudes)):
        raise ValueError("centers, widths, amplitudes must have the same length.")

    y = np.zeros_like(x_arr, dtype=float)
    for c, w, a in zip(centers, widths, amplitudes):
        w_eff = float(w) if float(w) > 0 else 1e-6
        y += float(a) * np.exp(-0.5 * ((x_arr - float(c)) / w_eff) ** 2)

    baseline = 0.15 * np.sin(2.0 * np.pi * (x_arr - xmin) / max(span, 1.0))
    y = y + baseline

    return standardize_signal(y) if standardize else y




SignalFn = Callable[..., np.ndarray]

SIGNALS: Dict[str, SignalFn] = {
    "smooth_trend": smooth_trend,
    "nonlinear_curvature": nonlinear_curvature,
    "local_bump": local_bump,
}


def get_signal(name: str) -> SignalFn:

    try:
        return SIGNALS[name]
    except KeyError as e:
        raise KeyError(f"Unknown signal '{name}'. Available: {list(SIGNALS.keys())}") from e


def generate_signal(
    x: ArrayLike,
    name: str,
    **kwargs
) -> np.ndarray:

    fn = get_signal(name)
    return fn(x, **kwargs)
