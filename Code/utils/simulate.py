from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, Union, Sequence
import numpy as np
from utils.signals import generate_signal, SIGNALS
from utils.errors import get_error_model, ERROR_MODELS


ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class SimulationResult:
    x: np.ndarray
    f: np.ndarray
    eps: np.ndarray
    y: np.ndarray
    meta: Dict[str, Any]


def _resolve_rng(seed: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    if rng is not None:
        return rng
    if seed is not None:
        return np.random.default_rng(seed)
    return np.random.default_rng()


def simulate_series(
    n: int,
    f_id: str,
    error_model: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    x_kind: str = "time",
    standardize_signal: bool = True,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> SimulationResult:

    if n <= 0:
        raise ValueError("n must be positive.")
    if f_id not in SIGNALS:
        raise KeyError(f"Unknown f_id '{f_id}'. Available: {list(SIGNALS.keys())}")
    if error_model not in ERROR_MODELS:
        raise KeyError(f"Unknown error_model '{error_model}'. Available: {list(ERROR_MODELS.keys())}")

    params = dict(params or {})
    rng = _resolve_rng(seed=seed, rng=rng)

    if x_kind == "time":
        x = np.arange(n, dtype=float)
    elif x_kind == "unit":
        x = np.linspace(0.0, 1.0, n)
    elif x_kind == "custom_grid":
        x_grid = params.pop("x_grid", None)
        if x_grid is None:
            raise ValueError("x_kind='custom_grid' requires params['x_grid'].")
        x = np.asarray(x_grid, dtype=float)
        if len(x) != n:
            raise ValueError("Length of x_grid must match n.")
    else:
        raise ValueError("x_kind must be one of: 'time', 'unit', 'custom_grid'.")

    f = generate_signal(x, f_id, standardize=standardize_signal)

    err_fn = get_error_model(error_model)
    eps = err_fn(n=n, rng=rng, **params)

    y = f + eps

    meta = {
        "n": n,
        "f_id": f_id,
        "error_model": error_model,
        "error_params": params,
        "x_kind": x_kind,
        "standardize_signal": standardize_signal,
    }

    return SimulationResult(x=x, f=f, eps=eps, y=y, meta=meta)


def generate_independent_test_set(
    n: int,
    f_id: str,
    error_model: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    x_kind: str = "time",
    standardize_signal: bool = True,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> SimulationResult:

    return simulate_series(
        n=n,
        f_id=f_id,
        error_model=error_model,
        params=params,
        x_kind=x_kind,
        standardize_signal=standardize_signal,
        seed=seed,
        rng=rng,
    )


def simulate_train_test_pair(
    n_train: int,
    n_test: int,
    f_id: str,
    error_model: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    x_kind: str = "time",
    standardize_signal: bool = True,
    seed: Optional[int] = None,
) -> tuple[SimulationResult, SimulationResult]:

    rng_master = _resolve_rng(seed=seed)

    s1, s2 = rng_master.integers(0, 2**32 - 1, size=2, dtype=np.uint32)
    train = simulate_series(
        n=n_train, f_id=f_id, error_model=error_model, params=params,
        x_kind=x_kind, standardize_signal=standardize_signal, seed=int(s1)
    )
    test = generate_independent_test_set(
        n=n_test, f_id=f_id, error_model=error_model, params=params,
        x_kind=x_kind, standardize_signal=standardize_signal, seed=int(s2)
    )
    return train, test
