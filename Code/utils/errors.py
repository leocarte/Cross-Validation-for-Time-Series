from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Union, Any
import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]




def empirical_acf(x: ArrayLike, nlags: int = 40, demean: bool = True) -> np.ndarray:

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("empirical_acf expects a 1D array.")
    n = len(x)
    if n == 0:
        raise ValueError("Empty input.")
    if nlags < 0:
        raise ValueError("nlags must be >= 0.")

    if demean:
        x = x - np.mean(x)

    denom = np.dot(x, x)
    if denom == 0:
        out = np.zeros(nlags + 1)
        out[0] = 1.0
        return out

    out = np.empty(nlags + 1, dtype=float)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        if k >= n:
            out[k] = np.nan
        else:
            out[k] = np.dot(x[:-k], x[k:]) / denom
    return out




def ar1_errors(
    n: int,
    rho: float,
    sigma: float = 1.0,
    burnin: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:

    if n <= 0:
        raise ValueError("n must be positive.")
    if abs(rho) >= 1:
        raise ValueError("For stationarity, require |rho| < 1.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if burnin < 0:
        raise ValueError("burnin must be >= 0.")

    rng = rng or np.random.default_rng()
    eta = rng.normal(0.0, sigma, size=n + burnin)

    e = np.zeros(n + burnin, dtype=float)
    for t in range(1, n + burnin):
        e[t] = rho * e[t - 1] + eta[t]

    return e[burnin:]


def ma5_errors(
    n: int,
    theta: Union[float, Sequence[float]] = 0.7,
    q: int = 5,
    sigma: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:

    if n <= 0:
        raise ValueError("n must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if q <= 0:
        raise ValueError("q must be positive.")

    if isinstance(theta, (int, float)):
        theta_vec = [float(theta)] * q
    else:
        theta_vec = list(theta)
        q = len(theta_vec)

    rng = rng or np.random.default_rng()
    eta = rng.normal(0.0, sigma, size=n + q)

    e = np.zeros(n, dtype=float)
    for t in range(n):
        idx = t + q - 1
        acc = eta[idx]
        for j, th in enumerate(theta_vec, start=1):
            acc += th * eta[idx - j]
        e[t] = acc

    return e


def arma_errors(
    n: int,
    ar: Sequence[float] = (),
    ma: Sequence[float] = (),
    sigma: float = 1.0,
    burnin: int = 300,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:

    if n <= 0:
        raise ValueError("n must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if burnin < 0:
        raise ValueError("burnin must be >= 0.")

    ar = list(ar)
    ma = list(ma)
    p = len(ar)
    q = len(ma)

    rng = rng or np.random.default_rng()
    total = n + burnin
    eta = rng.normal(0.0, sigma, size=total + q + 1)

    x = np.zeros(total, dtype=float)

    for t in range(total):
        ar_part = 0.0
        for i in range(1, p + 1):
            if t - i >= 0:
                ar_part += ar[i - 1] * x[t - i]

        ma_part = 0.0
        idx = t + q
        for j in range(1, q + 1):
            ma_part += ma[j - 1] * eta[idx - j]

        x[t] = ar_part + eta[idx] + ma_part

    return x[burnin:]


def arima_errors(
    n: int,
    p: int = 1,
    d: int = 0,
    q: int = 0,
    ar: Optional[Sequence[float]] = None,
    ma: Optional[Sequence[float]] = None,
    sigma: float = 1.0,
    burnin: int = 300,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:

    if n <= 0:
        raise ValueError("n must be positive.")
    if p < 0 or d < 0 or q < 0:
        raise ValueError("p, d, q must be >= 0.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    if ar is None:
        if p == 0:
            ar = []
        elif p == 1:
            ar = [0.5]
        else:
            ar = [0.3] * p

    if ma is None:
        if q == 0:
            ma = []
        elif q == 1:
            ma = [0.4]
        else:
            ma = [0.2] * q

    base = arma_errors(n=n, ar=ar, ma=ma, sigma=sigma, burnin=burnin, rng=rng)

    x = base
    for _ in range(d):
        x = np.cumsum(x)

    return x


def seasonal_ar_errors(
    n: int,
    season_period: int = 12,
    rho1: float = 0.3,
    rhoS: float = 0.5,
    sigma: float = 1.0,
    burnin: int = 300,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:

    if n <= 0:
        raise ValueError("n must be positive.")
    if season_period <= 0:
        raise ValueError("season_period must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if burnin < 0:
        raise ValueError("burnin must be >= 0.")

    rng = rng or np.random.default_rng()
    total = n + burnin
    eta = rng.normal(0.0, sigma, size=total)

    e = np.zeros(total, dtype=float)

    for t in range(total):
        val = eta[t]
        if t - 1 >= 0:
            val += rho1 * e[t - 1]
        if t - season_period >= 0:
            val += rhoS * e[t - season_period]
        e[t] = val

    return e[burnin:]




ErrorFn = Callable[..., np.ndarray]

ERROR_MODELS: Dict[str, ErrorFn] = {
    "ar1": ar1_errors,
    "ma5": ma5_errors,
    "arima": arima_errors,
    "seasonal_ar": seasonal_ar_errors,
}


def get_error_model(name: str) -> ErrorFn:
    try:
        return ERROR_MODELS[name]
    except KeyError as e:
        raise KeyError(f"Unknown error model '{name}'. Available: {list(ERROR_MODELS.keys())}") from e
