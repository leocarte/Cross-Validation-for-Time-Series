from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Sequence, Dict, Any
import random
import numpy as np



@dataclass(frozen=True)
class RNGManager:

    seed: int

    def seed_all(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)

    def generator(self, stream: int = 0) -> np.random.Generator:
        ss = np.random.SeedSequence(self.seed)
        child_seeds = ss.spawn(stream + 1)
        return np.random.default_rng(child_seeds[stream])


def seed_everything(seed: int) -> RNGManager:
    mgr = RNGManager(seed)
    mgr.seed_all()
    return mgr



def mse(y_true: Union[np.ndarray, Sequence[float]],
        y_pred: Union[np.ndarray, Sequence[float]]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"MSE shape mismatch: {yt.shape} vs {yp.shape}")
    err = yt - yp
    return float(np.mean(err * err))


def mae(y_true: Union[np.ndarray, Sequence[float]],
        y_pred: Union[np.ndarray, Sequence[float]]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"MAE shape mismatch: {yt.shape} vs {yp.shape}")
    return float(np.mean(np.abs(yt - yp)))


def metric_dict(y_true, y_pred) -> Dict[str, float]:
    return {"mse": mse(y_true, y_pred), "mae": mae(y_true, y_pred)}



@dataclass(frozen=True)
class TimeSeriesSplitSpec:

    train_size: float = 0.7
    test_size: float = 0.3
    gap: int = 0

    def __post_init__(self):
        total = self.train_size + self.test_size
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError(f"train_size + test_size must sum to 1.0. Got {total}.")
        if self.gap < 0:
            raise ValueError("gap must be >= 0.")
        if self.train_size <= 0 or self.test_size <= 0:
            raise ValueError("train_size and test_size must be > 0.")


def split_time_series_indices(
    n: int,
    spec: TimeSeriesSplitSpec = TimeSeriesSplitSpec()
) -> Tuple[np.ndarray, np.ndarray]:

    if n <= 0:
        raise ValueError("n must be positive.")

    gap = spec.gap

    n_train = int(np.floor(n * spec.train_size))
    n_test = n - n_train

    if n_train <= 0 or n_test <= 0:
        raise ValueError(
            f"Split too small for n={n}: train={n_train}, test={n_test}"
        )

    train_start = 0
    train_end = n_train

    test_start = train_end + gap
    test_end = n

    if test_start >= n:
        raise ValueError("gap too large: test split invalid or empty.")

    train_idx = np.arange(train_start, train_end)
    test_idx = np.arange(test_start, test_end)

    if len(test_idx) == 0:
        raise ValueError("Resulting test split empty due to gap or sizes.")

    return train_idx, test_idx


def split_time_series_arrays(
    X: Union[np.ndarray, Sequence[Any]],
    y: Union[np.ndarray, Sequence[Any]],
    spec: TimeSeriesSplitSpec = TimeSeriesSplitSpec()
):

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if len(X_arr) != len(y_arr):
        raise ValueError("X and y must have the same length.")

    train_idx, test_idx = split_time_series_indices(len(X_arr), spec)

    return (
        (X_arr[train_idx], y_arr[train_idx]),
        (X_arr[test_idx], y_arr[test_idx]),
    )


def describe_split(
    n: int,
    spec: TimeSeriesSplitSpec = TimeSeriesSplitSpec()
) -> Dict[str, Any]:

    tr, te = split_time_series_indices(n, spec)
    return {
        "n": n,
        "gap": spec.gap,
        "train_size": len(tr),
        "test_size": len(te),
        "train_range": (int(tr[0]), int(tr[-1])) if len(tr) else None,
        "test_range": (int(te[0]), int(te[-1])) if len(te) else None,
    }