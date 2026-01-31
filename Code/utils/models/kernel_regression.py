from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

from utils.models.base import BaseRegressor


KernelType = Literal["gaussian", "epanechnikov"]


def _gaussian(u: np.ndarray) -> np.ndarray:
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        return np.exp(-0.5 * u * u)


def _epanechnikov(u: np.ndarray) -> np.ndarray:
    out = 1.0 - u * u
    out[out < 0.0] = 0.0
    return out


@dataclass
class KernelRegressor(BaseRegressor):

    param_value: float = 1.0
    kernel: KernelType = "gaussian"
    ridge: float = 1e-12

    _x_train: Optional[np.ndarray] = None
    _y_train: Optional[np.ndarray] = None

    def __post_init__(self):
        self.model_id = "kernel_regression"
        self.param_name = "h"

        if self.param_value <= 0:
            raise ValueError("Bandwidth h must be > 0.")

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> "KernelRegressor":
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        self._x_train = x
        self._y_train = y
        return self

    def _kernel_eval(self, u: np.ndarray) -> np.ndarray:
        if self.kernel == "gaussian":
            return _gaussian(u)
        elif self.kernel == "epanechnikov":
            return _epanechnikov(u)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def predict(self, x: np.ndarray, *, batch: int = 256) -> np.ndarray:
        if self._x_train is None or self._y_train is None:
            raise RuntimeError("Model is not fitted.")

        xq = np.asarray(x, dtype=np.float32).reshape(-1)
        xt = self._x_train
        yt = self._y_train
        h = float(self.param_value)
        n_q = len(xq)

        out = np.empty(n_q, dtype=np.float32)
        for start in range(0, n_q, batch):
            end = min(start + batch, n_q)
            diffs = (xq[start:end, None] - xt[None, :]) / h
            w = self._kernel_eval(diffs)
            with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
                w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                num = w @ yt
                den = np.sum(w, axis=1) + self.ridge
            out[start:end] = num / den

        return out

    def clone(self) -> "KernelRegressor":
        return KernelRegressor(
            param_value=float(self.param_value),
            kernel=self.kernel,
            ridge=self.ridge,
        )
