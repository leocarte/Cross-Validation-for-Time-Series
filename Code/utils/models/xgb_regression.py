from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from xgboost import XGBRegressor

from utils.models.base import BaseRegressor


@dataclass
class XGBTreeRegressor(BaseRegressor):

    param_value: int = 3

    n_estimators: int = 200
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    n_jobs: int = 1

    _model: Optional[XGBRegressor] = None

    def __post_init__(self):
        self.model_id = "xgb_tree"
        self.param_name = "max_depth"
        self._validate_depth(self.param_value)

    def _validate_depth(self, d: int) -> None:
        if not np.isfinite(d):
            raise ValueError("max_depth must be finite.")
        d = int(d)
        if d <= 0:
            raise ValueError("max_depth must be > 0.")

    def set_param(self, value):
        d = int(value)
        self._validate_depth(d)
        self.param_value = d


    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> "XGBTreeRegressor":
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        if len(x) == 0:
            raise ValueError("Empty training data.")

        X = x.reshape(-1, 1)

        self._model = XGBRegressor(
            max_depth=int(self.param_value),
            n_estimators=int(self.n_estimators),
            learning_rate=float(self.learning_rate),
            subsample=float(self.subsample),
            colsample_bytree=float(self.colsample_bytree),
            reg_lambda=float(self.reg_lambda),
            min_child_weight=float(self.min_child_weight),
            objective="reg:squarederror",
            random_state=0,
            n_jobs=self.n_jobs,
            verbosity=0,
        )

        self._model.fit(X, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted.")

        x = np.asarray(x, dtype=float).reshape(-1)
        X = x.reshape(-1, 1)
        yhat = self._model.predict(X)
        return np.asarray(yhat, dtype=float).reshape(-1)

    def clone(self) -> "XGBTreeRegressor":
        return XGBTreeRegressor(
            param_value=int(self.param_value),
            n_estimators=int(self.n_estimators),
            learning_rate=float(self.learning_rate),
            subsample=float(self.subsample),
            colsample_bytree=float(self.colsample_bytree),
            reg_lambda=float(self.reg_lambda),
            min_child_weight=float(self.min_child_weight),
            n_jobs=int(self.n_jobs),
        )
