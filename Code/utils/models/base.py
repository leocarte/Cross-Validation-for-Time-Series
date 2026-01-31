from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import numpy as np


class Regressor(Protocol):
    model_id: str
    param_name: str
    param_value: Any

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> "Regressor":
        ...

    def predict(self, x: np.ndarray) -> np.ndarray:
        ...

    def set_param(self, value: Any) -> None:
        ...

    def get_param(self) -> Any:
        ...

    def clone(self) -> "Regressor":
        ...


@dataclass
class BaseRegressor:
    param_value: Any

    model_id: str = field(init=False, default="")
    param_name: str = field(init=False, default="")

    def set_param(self, value: Any) -> None:
        self.param_value = value

    def get_param(self) -> Any:
        return self.param_value

    def fit(self, x, y, **kwargs):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def clone(self):
        return type(self)(param_value=self.param_value)
