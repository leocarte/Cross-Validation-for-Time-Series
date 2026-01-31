from __future__ import annotations

from typing import Sequence, Union, Optional, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt


from utils.errors import ERROR_MODELS
from utils.simulate import simulate_series


ArrayLike = Union[np.ndarray, Sequence[float]]



DEFAULT_ERROR_PARAMS: Dict[str, Dict[str, Any]] = {
    "ar1": {"rho": 0.7, "sigma": 1.0},
    "ma5": {"theta": 0.7, "sigma": 1.0},
    "arima": {"p": 1, "d": 0, "q": 1, "ar": [0.6], "ma": [0.4], "sigma": 1.0},
    "seasonal_ar": {"season_period": 12, "rho1": 0.2, "rhoS": 0.6, "sigma": 1.0},
}



def empirical_acf(
    x: ArrayLike,
    nlags: int = 40,
    demean: bool = True
) -> np.ndarray:

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

    denom = float(np.dot(x, x))
    if denom == 0.0:
        out = np.zeros(nlags + 1, dtype=float)
        out[0] = 1.0
        return out

    out = np.empty(nlags + 1, dtype=float)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        if k >= n:
            out[k] = np.nan
        else:
            out[k] = float(np.dot(x[:-k], x[k:]) / denom)

    return out


def acf_confidence_bounds(n: int, alpha: float = 0.05) -> Tuple[float, float]:

    if n <= 0:
        raise ValueError("n must be positive.")

    z_map = {0.10: 1.645, 0.05: 1.96, 0.01: 2.576}
    z = z_map.get(alpha, 1.96)

    bound = z / np.sqrt(n)
    return -bound, bound



def plot_acf(
    series: ArrayLike,
    *,
    nlags: int = 40,
    title: str = "ACF",
    demean: bool = True,
    show_confidence: bool = True,
    alpha: float = 0.05,
    show: bool = True,
    save_path: Optional[str] = None,
):

    series = np.asarray(series, dtype=float)
    acf_vals = empirical_acf(series, nlags=nlags, demean=demean)
    lags = np.arange(len(acf_vals))

    plt.figure()
    plt.plot(lags, acf_vals)
    plt.axhline(0.0)

    if show_confidence:
        lo, hi = acf_confidence_bounds(len(series), alpha=alpha)
        plt.axhline(lo, linestyle="--")
        plt.axhline(hi, linestyle="--")

    plt.title(title)
    plt.xlabel("lag")
    plt.ylabel("ACF")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_all_error_acfs(
    *,
    n: int = 400,
    nlags: int = 40,
    params_map: Optional[Dict[str, Dict[str, Any]]] = None,
    demean: bool = True,
    show_confidence: bool = False,
    alpha: float = 0.05,
    show: bool = True,
    save_path: Optional[str] = None,
):

    params_map = params_map or DEFAULT_ERROR_PARAMS

    plt.figure()
    for name, fn in ERROR_MODELS.items():
        params = params_map.get(name, {})
        eps = fn(n=n, **params)
        acf_vals = empirical_acf(eps, nlags=nlags, demean=demean)

        lags = np.arange(len(acf_vals))
        plt.plot(lags, acf_vals, label=name)

    plt.title("Empirical ACF of synthetic error processes (εₜ)")
    plt.xlabel("lag")
    plt.ylabel("ACF")
    plt.axhline(0.0)

    if show_confidence:
        lo, hi = acf_confidence_bounds(n, alpha=alpha)
        plt.axhline(lo, linestyle="--")
        plt.axhline(hi, linestyle="--")

    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_each_error_acf(
    *,
    n: int = 400,
    nlags: int = 40,
    params_map: Optional[Dict[str, Dict[str, Any]]] = None,
    demean: bool = True,
    show_confidence: bool = True,
    alpha: float = 0.05,
    show: bool = True,
    save_dir: Optional[str] = None,
):

    params_map = params_map or DEFAULT_ERROR_PARAMS

    for name, fn in ERROR_MODELS.items():
        params = params_map.get(name, {})
        eps = fn(n=n, **params)

        save_path = f"{save_dir}/{name}_acf.png" if save_dir else None
        plot_acf(
            eps,
            nlags=nlags,
            title=f"Empirical ACF: {name} (εₜ)",
            demean=demean,
            show_confidence=show_confidence,
            alpha=alpha,
            show=show,
            save_path=save_path,
        )


def plot_acfs_for_full_series(
    *,
    n: int = 400,
    nlags: int = 40,
    f_id: str = "smooth_trend",
    params_map: Optional[Dict[str, Dict[str, Any]]] = None,
    demean: bool = True,
    show_confidence: bool = False,
    alpha: float = 0.05,
    show: bool = True,
    save_path: Optional[str] = None,
):

    params_map = params_map or DEFAULT_ERROR_PARAMS

    plt.figure()
    for name in ERROR_MODELS.keys():
        params = params_map.get(name, {})

        sim = simulate_series(
            n=n,
            f_id=f_id,
            error_model=name,
            params=params,
            x_kind="time",
            standardize_signal=True,
        )

        acf_vals = empirical_acf(sim.y, nlags=nlags, demean=demean)
        lags = np.arange(len(acf_vals))
        plt.plot(lags, acf_vals, label=name)

    plt.title(f"Empirical ACF of observed series yₜ (f_id={f_id})")
    plt.xlabel("lag")
    plt.ylabel("ACF")
    plt.axhline(0.0)

    if show_confidence:
        lo, hi = acf_confidence_bounds(n, alpha=alpha)
        plt.axhline(lo, linestyle="--")
        plt.axhline(hi, linestyle="--")

    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()