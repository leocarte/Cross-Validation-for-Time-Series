from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Callable, Iterable, Optional
import numpy as np

from utils.synthetic_series import SyntheticSeries
from utils.models.kernel_regression import KernelRegressor
from utils.models.penalized_spline import PenalizedSplineRegressor
from utils.models.base import Regressor
from utils.cv import (
    naive_kfold_splits,
    block_kfold_splits,
    leave_2l_plus_1_out_splits,
    walk_forward_splits,
    WalkForwardSpec,
    assert_disjoint,
    assert_buffer_excluded,
)
from utils.search import cv_select_param, mse




def kernel_factory() -> Regressor:
    return KernelRegressor(param_value=1.0, kernel="gaussian")

def spline_factory() -> Regressor:
    return PenalizedSplineRegressor(param_value=1.0, n_knots=30, degree=3)


MODEL_FACTORIES: Dict[str, Callable[[], Regressor]] = {
    "kernel": kernel_factory,
    "spline": spline_factory,
    # "xgb": xgb_factory,
}




def make_splits(cv_name: str, n: int, *, seed: int = 0, k: int = 5, l: int = 2) -> Iterable:
    if cv_name == "naive":
        return list(naive_kfold_splits(n, k=k, seed=seed))
    if cv_name == "block":
        return list(block_kfold_splits(n, k=k))
    if cv_name == "leave_2l_plus_1_out":
        return list(leave_2l_plus_1_out_splits(n, l=l))
    if cv_name == "walk_forward":
        spec = WalkForwardSpec(
            initial_train_size=max(10, n // 3),
            val_size=max(5, n // 10),
            step=max(1, n // 10),
        )
        return list(walk_forward_splits(n, spec))
    raise KeyError(f"Unknown CV scheme '{cv_name}'.")


CV_SCHEMES = ["naive", "block", "leave_2l_plus_1_out", "walk_forward"]




@dataclass(frozen=True)
class DGP:
    n: int
    f_id: str
    error_model: str
    error_params: Dict[str, Any]




def run_experiments(
    dgp_grid: List[DGP],
    *,
    model_names: List[str] = ["kernel"],
    cv_schemes: List[str] = CV_SCHEMES,
    h_grid: Optional[List[float]] = None,
    lambda_grid: Optional[List[float]] = None,
    k_folds: int = 5,
    buffer_l: int = 2,
    replications: int = 5,
    base_seed: int = 123,
) -> List[Dict[str, Any]]:

    if h_grid is None:
        h_grid = [0.2, 0.5, 1.0, 2.0, 5.0]
    
    if lambda_grid is None:
        lambda_grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    results: List[Dict[str, Any]] = []

    for dgp in dgp_grid:
        for rep in range(replications):
            seed_train = base_seed + rep * 10_000 + hash((dgp.f_id, dgp.error_model)) % 10_000
            seed_test = seed_train + 777

            train_sim = SyntheticSeries.generate(
                n=dgp.n,
                f_id=dgp.f_id,
                error_model=dgp.error_model,
                params=dgp.error_params,
                seed=seed_train,
            )
            test_sim = SyntheticSeries.generate(
                n=dgp.n,
                f_id=dgp.f_id,
                error_model=dgp.error_model,
                params=dgp.error_params,
                seed=seed_test,
            )

            x_train, y_train = train_sim.x, train_sim.y
            x_test, y_test = test_sim.x, test_sim.y

            n_train = len(x_train)

            for model_name in model_names:
                if model_name not in MODEL_FACTORIES:
                    raise KeyError(f"Unknown model '{model_name}'.")

                model_factory = MODEL_FACTORIES[model_name]

                if model_name == "kernel":
                    grid = h_grid
                elif model_name == "spline":
                    grid = lambda_grid
                else:
                    raise KeyError(f"No grid defined for model '{model_name}'.")

                for cv_name in cv_schemes:
                    splits = make_splits(cv_name, n_train, seed=seed_train, k=k_folds, l=buffer_l)

                    for tr, va in splits:
                        assert_disjoint(tr, va)
                        if cv_name == "buffered":
                            assert_buffer_excluded(tr, va, buffer_l, n_train)

                    cv_res = cv_select_param(
                        model_factory=model_factory,
                        x=x_train,
                        y=y_train,
                        splits=splits,
                        grid=grid,
                        metric=mse,
                    )

                    final_model = model_factory()
                    final_model.set_param(cv_res.best_param)
                    final_model.fit(x_train, y_train)

                    y_pred_test = final_model.predict(x_test)
                    test_err = mse(y_test, y_pred_test)

                    results.append({
                        "dgp_n": dgp.n,
                        "f_id": dgp.f_id,
                        "error_model": dgp.error_model,
                        "error_params": dict(dgp.error_params),
                        "replication": rep,
                        "seed_train": seed_train,
                        "seed_test": seed_test,
                        "model": model_name,
                        "cv_scheme": cv_name,
                        "param_name": getattr(final_model, "param_name", "param"),
                        "selected_param": cv_res.best_param,
                        "cv_est_error": cv_res.best_score,
                        "test_error": test_err,
                    })

    return results
