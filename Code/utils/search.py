from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Iterable, List, Tuple, Optional
import numpy as np

from utils.models.base import Regressor
from utils.cv import Split


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


@dataclass
class CVResult:
    best_param: Any
    best_score: float
    scores: Dict[Any, float]


def cv_select_param(
    model_factory: Callable[[], Regressor],
    x: np.ndarray,
    y: np.ndarray,
    splits: Iterable[Split],
    grid: Iterable[Any],
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] = mse,
) -> CVResult:

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    grid = list(grid)
    scores: Dict[Any, float] = {}

    for param in grid:
        fold_scores: List[float] = []
        for train_idx, val_idx in splits:
            model = model_factory()
            model.set_param(param)

            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[val_idx])
            fold_scores.append(metric(y[val_idx], pred))

        scores[param] = float(np.mean(fold_scores))

    best_param = min(scores, key=scores.get)
    return CVResult(best_param=best_param, best_score=scores[best_param], scores=scores)


def fast_ncv_select_param(
    model_factory: Callable[[], Regressor],
    x: np.ndarray,
    y: np.ndarray,
    grid: Iterable[Any],
    l_buffer: int,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] = mse,
) -> CVResult:

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    grid = list(grid)
    scores: Dict[Any, float] = {}

    for param in grid:
        model = model_factory()
        if not hasattr(model, "compute_ncv_score"):
            raise AttributeError("Model does not support fast NCV (missing compute_ncv_score)")
        model.set_param(param)
        model.fit(x, y)
        scores[param] = float(model.compute_ncv_score(param, l_buffer))

    best_param = min(scores, key=scores.get)
    return CVResult(best_param=best_param, best_score=scores[best_param], scores=scores)
