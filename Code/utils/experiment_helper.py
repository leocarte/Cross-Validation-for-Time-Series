from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple, Iterable, Union, Set
from collections import defaultdict
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


try:
    from utils.synthetic_series import SyntheticSeries
    from utils.search import cv_select_param, fast_ncv_select_param, mse
    from utils.cv import (
        naive_kfold_splits,
        block_kfold_splits,
        block_with_buffer_splits,
        leave_2l_plus_1_out_splits,
        walk_forward_splits,
        WalkForwardSpec,
        assert_disjoint,
        assert_buffer_excluded,
    )
except ImportError:
    pass 



@dataclass(frozen=True)
class DGP:
    n: int
    f_id: str
    error_model: str
    error_params: Dict[str, Any]


@dataclass
class ModelSpec:

    name: str
    factory: Callable[[], Any]
    grid: List[Any]




Split = Tuple[np.ndarray, np.ndarray]


def make_splits(
    cv_name: str,
    n: int,
    *,
    seed: int = 0,
    k: int = 5,
    l: int = 2,
    walk_spec: Optional[WalkForwardSpec] = None,
) -> List[Split]:
    if cv_name == "naive":
        return list(naive_kfold_splits(n, k=k, seed=seed))
    if cv_name == "block":
        return list(block_kfold_splits(n, k=k))
    if cv_name == "block_buffered":
        return list(block_with_buffer_splits(n, k=k, l=l))
    if cv_name == "leave_2l_plus_1_out":
        return list(leave_2l_plus_1_out_splits(n, l=l))
    if cv_name == "walk_forward":
        if walk_spec is None:
            walk_spec = WalkForwardSpec(
                initial_train_size=max(20, n // 3),
                val_size=max(10, n // 10),
                step=max(10, n // 10),
            )
        return list(walk_forward_splits(n, walk_spec))
    raise KeyError(f"Unknown CV scheme '{cv_name}'.")




def plot_cv_boxplots_grid(
    df: pd.DataFrame,
    *,
    metrics: Tuple[str, ...] = ("test_error", "cv_est_error", "cv_bias"),
    models_order: Optional[List[str]] = None,
    cv_order: Optional[List[str]] = None,
    title: str = "CV Performance Comparison",
    yscale: Optional[str] = None,
    yscale_per_metric: Optional[Dict[str, str]] = None,
    save_path_png: Optional[str] = None,
    save_path_pdf: Optional[str] = None,
):

    df_plot = df.copy()
    
    name_map = {
        'naive': 'Naive', 
        'block': 'Block', 
        'leave_2l_plus_1_out': 'Buffered', 
        'block_buffered': 'Buffered', 
        'walk_forward': 'Walk-Forward',
        'window': 'Wind.'
    }
    
    if 'cv_scheme' in df_plot.columns:
        df_plot['cv_short'] = df_plot['cv_scheme'].map(name_map).fillna(df_plot['cv_scheme'])
    else:
        df_plot['cv_short'] = df_plot['cv_scheme_display']
        
    if models_order is None:
        models_order = sorted(df_plot["model"].unique())
        
    if cv_order is None:
        preferred = ["Naive", "Block", "Buffered", "Walk-Forward"]
        existing = df_plot['cv_short'].unique()
        cv_order_mapped = [c for c in preferred if c in existing]
        for c in existing:
            if c not in cv_order_mapped:
                cv_order_mapped.append(c)
    else:
        cv_order_mapped = [name_map.get(c, c) for c in cv_order]

    valid_metrics = [m for m in metrics if m in df_plot.columns and df_plot[m].notna().any()]
    if not valid_metrics:
        print("No valid metrics found to plot.")
        return

    n_rows = len(models_order)
    n_cols = len(valid_metrics)
    

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(13.2, 2.2 * n_rows),
        sharey="col", 
        sharex=True,
        constrained_layout=True
    )

    axes = np.atleast_2d(axes)
    if n_rows == 1 and n_cols > 1: axes = axes.reshape(1, -1)
    if n_cols == 1 and n_rows > 1: axes = axes.reshape(-1, 1)

    for i, model in enumerate(models_order):
        sub_m = df_plot[df_plot["model"] == model]

        for j, metric in enumerate(valid_metrics):
            ax = axes[i, j]
            
            sns.boxplot(
                data=sub_m,
                x='cv_short',
                y=metric,
                hue='cv_short', 
                order=cv_order_mapped,
                palette='Set2',
                width=0.6,
                linewidth=1, 
                showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"white", 
                           "markeredgecolor":"black", "markersize": 3},
                legend=False, 
                ax=ax
            )

            ax.grid(axis='y', linestyle=':', alpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_xlabel("") 
            ax.set_ylabel("") 

            if j == 0:
                ax.set_ylabel(model, fontsize=9, fontweight='bold')

            metric_scale = None
            if yscale_per_metric:
                metric_scale = yscale_per_metric.get(metric)
            if metric_scale is None:
                metric_scale = yscale

            if i == 0:
                clean_metric = metric.replace('_', ' ').title()
                if "Est Error" in clean_metric: clean_metric = "CV Est. Error"
                if "Test Error" in clean_metric: clean_metric = "Test Error"
                if "Cv Bias" in clean_metric: clean_metric = "CV Bias"
                if metric_scale == "log" and metric == "test_error":
                    clean_metric = "Test Error (log)"
                if metric_scale == "log" and metric == "cv_est_error":
                    clean_metric = "CV Est. Error (log)"
                
                ax.set_title(clean_metric, fontsize=10, fontweight='bold')
            if metric_scale:
                ax.set_yscale(metric_scale)

    fig.suptitle(title, fontsize=11, fontweight='bold')
    
    if save_path_png:
        fig.savefig(save_path_png, bbox_inches="tight", dpi=300)
    if save_path_pdf:
        fig.savefig(save_path_pdf, bbox_inches="tight")
        
    plt.show()
    return df



def plot_fits_grid_one_rep(
    model_specs: Dict[str, ModelSpec],
    cv_schemes: List[str],
    *,
    dgp: Optional[DGP] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    f_true: Optional[np.ndarray] = None,
    seed: int = 123,
    k_folds: int = 5,
    buffer_l: int = 2,
    walk_spec: Optional[WalkForwardSpec] = None,
):

    if dgp is not None:
        sim = SyntheticSeries.generate(
            n=dgp.n,
            f_id=dgp.f_id,
            error_model=dgp.error_model,
            params=dgp.error_params,
            seed=seed,
        )
        x_data = sim.x
        y_data = sim.y
        f_curve = sim.f
    else:
        if x is None or y is None:
            raise ValueError("If DGP is not provided, x and y must be supplied.")
        x_data = np.asarray(x)
        y_data = np.asarray(y)
        f_curve = f_true if f_true is not None else None

    n_train = len(x_data)
    model_list = list(model_specs.values())
    n_rows = len(model_list)
    n_cols = len(cv_schemes)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.2 * n_rows),
        sharex=True, sharey=True
    )

    axes = np.asarray(axes)
    if n_rows == 1 and n_cols == 1:
        axes = axes.reshape(1, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, mspec in enumerate(model_list):
        for j, cv_name in enumerate(cv_schemes):
            ax = axes[i, j]

            splits = make_splits(
                cv_name,
                n_train,
                seed=seed,
                k=k_folds,
                l=buffer_l,
                walk_spec=walk_spec,
            )

            cv_res = cv_select_param(
                model_factory=mspec.factory,
                x=x_data,
                y=y_data,
                splits=splits,
                grid=mspec.grid,
                metric=mse,
            )

            model = mspec.factory()
            model.set_param(cv_res.best_param)
            model.fit(x_data, y_data)
            yhat = model.predict(x_data)

            pname = getattr(model, "param_name", "param")

            ax.plot(x_data, y_data, alpha=0.5, label="observed y", color="gray")
            
            if f_curve is not None:
                ax.plot(x_data, f_curve, linewidth=2, label="true f", color="black", linestyle="--")
            
            ax.plot(x_data, yhat, linewidth=2, label="fitted", color="tab:blue")

            if i == 0:
                ax.set_title(cv_name)
            if j == 0:
                ax.set_ylabel(mspec.name)

            ax.text(
                0.02, 0.95,
                f"{pname}={cv_res.best_param}\nCV MSE={cv_res.best_score:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle(
        "One-rep fitted curves by model (rows) and CV scheme (cols)",
        y=1.02, 
        fontweight='bold'
    )
    fig.tight_layout()
    plt.show()


def plot_fits_one_rep(
    model_specs: Dict[str, ModelSpec],
    cv_schemes: List[str],
    *,
    dgp: Optional[DGP] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    f_true: Optional[np.ndarray] = None,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    f_test_true: Optional[np.ndarray] = None,
    seed: int = 123,
    k_folds: int = 5,
    buffer_l: int = 2,
    walk_spec: Optional[WalkForwardSpec] = None,
    show_test: bool = False,
):

    if dgp is not None:
        train_sim = SyntheticSeries.generate(
            n=dgp.n,
            f_id=dgp.f_id,
            error_model=dgp.error_model,
            params=dgp.error_params,
            seed=seed,
        )
        x_tr, y_tr, f_tr = train_sim.x, train_sim.y, train_sim.f
    else:
        if x is None or y is None:
            raise ValueError("If DGP is not provided, x and y must be supplied.")
        x_tr, y_tr = np.asarray(x), np.asarray(y)
        f_tr = f_true  

    n_train = len(x_tr)

    x_te, y_te, f_te = None, None, None
    if show_test:
        if dgp is not None:
            test_sim = SyntheticSeries.generate(
                n=dgp.n,
                f_id=dgp.f_id,
                error_model=dgp.error_model,
                params=dgp.error_params,
                seed=seed + 777,
            )
            x_te, y_te, f_te = test_sim.x, test_sim.y, test_sim.f
        else:
            if x_test is None or y_test is None:
                print("Warning: show_test=True but no test data provided. Skipping test plots.")
                show_test = False
            else:
                x_te, y_te = np.asarray(x_test), np.asarray(y_test)
                f_te = f_test_true

    for _, mspec in model_specs.items():
        fitted: Dict[str, Dict[str, Any]] = {}

        for cv_name in cv_schemes:
            splits = make_splits(
                cv_name,
                n_train,
                seed=seed,
                k=k_folds,
                l=buffer_l,
                walk_spec=walk_spec,
            )

            cv_res = cv_select_param(
                model_factory=mspec.factory,
                x=x_tr,
                y=y_tr,
                splits=splits,
                grid=mspec.grid,
                metric=mse,
            )

            model = mspec.factory()
            model.set_param(cv_res.best_param)
            model.fit(x_tr, y_tr)

            yhat_train = model.predict(x_tr)
            fitted[cv_name] = {
                "best_param": cv_res.best_param,
                "cv_err": cv_res.best_score,
                "yhat_train": yhat_train,
                "param_name": getattr(model, "param_name", "param"),
            }

            if show_test and x_te is not None:
                fitted[cv_name]["yhat_test"] = model.predict(x_te)

        if not show_test:
            cols = 2
            rows = int(np.ceil(len(cv_schemes) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(12, 4.5 * rows), sharex=True, sharey=True)
            axes = np.asarray(axes).ravel()

            for ax, cv_name in zip(axes, cv_schemes):
                info = fitted[cv_name]
                pname = info["param_name"]

                ax.plot(x_tr, y_tr, label="train y", alpha=0.6)
                if f_tr is not None:
                    ax.plot(x_tr, f_tr, label="true f", linewidth=2)
                ax.plot(x_tr, info["yhat_train"], label="fitted", linewidth=2)

                ax.set_title(f"{cv_name} | {pname}={info['best_param']} | CV MSE={info['cv_err']:.3f}")
                ax.set_xlabel("t")
                ax.set_ylabel("value")
                ax.legend()

            for k in range(len(cv_schemes), len(axes)):
                axes[k].axis("off")

            fig.suptitle(f"One-rep fits for model: {mspec.name}", y=1.02)
            fig.tight_layout()
            plt.show()

        else:
            rows = len(cv_schemes)
            fig, axes = plt.subplots(rows, 2, figsize=(12, 3.2 * rows), sharex=False, sharey=True)

            if rows == 1:
                axes = np.array([axes])

            for i, cv_name in enumerate(cv_schemes):
                info = fitted[cv_name]
                pname = info["param_name"]

                ax_tr = axes[i, 0]
                ax_te = axes[i, 1]

                ax_tr.plot(x_tr, y_tr, label="train y", alpha=0.6)
                if f_tr is not None:
                    ax_tr.plot(x_tr, f_tr, label="true f", linewidth=2)
                ax_tr.plot(x_tr, info["yhat_train"], label="fitted", linewidth=2)
                ax_tr.set_title(f"{cv_name} (train) | {pname}={info['best_param']}")

                ax_te.plot(x_te, y_te, label="test y", alpha=0.6)
                if f_te is not None:
                    ax_te.plot(x_te, f_te, label="true f", linewidth=2)
                ax_te.plot(x_te, info["yhat_test"], label="train-fit pred", linewidth=2)
                ax_te.set_title(f"{cv_name} (test)")

                if i == rows - 1:
                    ax_tr.set_xlabel("t")
                    ax_te.set_xlabel("t")

                ax_tr.set_ylabel("value")
                ax_tr.legend()
                ax_te.legend()

            fig.suptitle(f"Train vs Test fits for model: {mspec.name}", y=1.01, fontweight='bold')
            fig.tight_layout()
            plt.show()




def run_cv_benchmark(
    model_specs: Dict[str, ModelSpec],
    cv_schemes: List[str],
    *,
    dgp: Optional[DGP] = None,
    x_train_fixed: Optional[np.ndarray] = None,
    y_train_fixed: Optional[np.ndarray] = None,
    x_test_fixed: Optional[np.ndarray] = None,
    y_test_fixed: Optional[np.ndarray] = None,
    replications: int = 100,
    base_seed: int = 123,
    k_folds: int = 5,
    buffer_l: int = 2,
    walk_spec: Optional[WalkForwardSpec] = None,
    rolling_test: bool = False,
    ahead_h: int = 1,
    ahead_l: int = 0,
    rolling_test_strategy: str = "expanding", 
    rolling_window: Optional[int] = None,
    verbose: bool = False,  
    return_predictions: bool = False,
    skip_replications: Optional[Dict[str, Set[int]]] = None,
    save_after_each_replication: bool = False,
    run_dir: Optional[Path] = None,
    persist_format: str = "parquet",
) -> pd.DataFrame:

    if ahead_h < 1:
        raise ValueError("ahead_h must be >= 1")
    if ahead_l < 0:
        raise ValueError("ahead_l must be >= 0")

    def _select_param(mspec, cv_name, x_arr, y_arr, *, seed, k_folds, buffer_l, walk_spec):

        if cv_name == "block_buffered":
            probe = mspec.factory()
            if hasattr(probe, "compute_ncv_score"):
                return fast_ncv_select_param(
                    model_factory=mspec.factory,
                    x=x_arr,
                    y=y_arr,
                    grid=mspec.grid,
                    l_buffer=buffer_l,
                )

        splits = make_splits(
            cv_name,
            len(x_arr),
            seed=seed,
            k=k_folds,
            l=buffer_l,
            walk_spec=walk_spec,
        )

        for tr_idx, va_idx in splits:
            assert_disjoint(tr_idx, va_idx)
            if cv_name == "leave_2l_plus_1_out_splits":
                assert_buffer_excluded(tr_idx, va_idx, buffer_l, len(x_arr))

        return cv_select_param(
            model_factory=mspec.factory,
            x=x_arr,
            y=y_arr,
            splits=splits,
            grid=mspec.grid,
            metric=mse,
        )

    rows: List[Dict[str, Any]] = []
    if skip_replications is None:
        skip_replications = {}
    skip_replications = {k: set(v) for k, v in skip_replications.items()}

    predictions_store: Optional[Dict[str, List[Dict[str, Any]]]] = None
    if return_predictions:
        predictions_store = {k: [] for k in model_specs}

    if dgp is None and (x_train_fixed is None or y_train_fixed is None):
        raise ValueError("Must provide either 'dgp' OR 'x_train_fixed' and 'y_train_fixed'.")

    start_time = time.time()
    
    if verbose:
        print(f"Starting Benchmark: {replications} Replications...")
        print("-" * 60)

    for rep in range(replications):
        if verbose:
            elapsed = time.time() - start_time
            if rep > 0:
                avg_time_per_rep = elapsed / rep
                remaining_reps = replications - rep
                est_remaining = avg_time_per_rep * remaining_reps
                time_str = f"Elapsed: {elapsed:.1f}s | ETA: {est_remaining:.1f}s"
            else:
                time_str = f"Elapsed: {elapsed:.1f}s | ETA: --"
            
            print(f"Replication {rep + 1}/{replications} | {time_str}")

        seed_train = base_seed + rep * 1000 + 17
        seed_test = seed_train + 777

        if dgp is not None:
            sim = SyntheticSeries.generate(
                n=dgp.n,
                f_id=dgp.f_id,
                error_model=dgp.error_model,
                params=dgp.error_params,
                seed=seed_train,
            )

            split_idx = int(0.7 * len(sim.x))
            x_tr, y_tr = sim.x[:split_idx], sim.y[:split_idx]
            x_te, y_te = sim.x[split_idx:], sim.y[split_idx:]
            f_tr = sim.f[:split_idx]
            f_te = sim.f[split_idx:]
            
            dgp_info = {
                "dgp_n": dgp.n,
                "f_id": dgp.f_id,
                "error_model": dgp.error_model,
                "error_params": dict(dgp.error_params),
            }
        else:
            x_tr, y_tr = x_train_fixed, y_train_fixed
            x_te, y_te = x_test_fixed, y_test_fixed
            f_tr, f_te = None, None
            
            dgp_info = {
                "dgp_n": len(x_tr),
                "f_id": "real_data",
                "error_model": "real_data",
                "error_params": {},
            }

        n_train = len(x_tr)

        for mkey, mspec in model_specs.items():
            if rep in skip_replications.get(mkey, set()):
                continue

            rep_capture: Optional[Dict[str, Any]] = None
            if predictions_store is not None:
                rep_capture = {
                    "replication": rep,
                    "seed_train": seed_train,
                    "seed_test": seed_test,
                    "x_train": np.asarray(x_tr),
                    "y_train": np.asarray(y_tr),
                    "f_train": np.asarray(f_tr) if f_tr is not None else None,
                    "x_test": np.asarray(x_te) if x_te is not None else None,
                    "y_test": np.asarray(y_te) if y_te is not None else None,
                    "f_test": np.asarray(f_te) if f_te is not None else None,
                    "cv_results": {},
                }
            for cv_name in cv_schemes:
                
                cv_start = time.time()
                cv_res = _select_param(
                    mspec,
                    cv_name,
                    x_tr,
                    y_tr,
                    seed=seed_train,
                    k_folds=k_folds,
                    buffer_l=buffer_l,
                    walk_spec=walk_spec,
                )
                cv_time_seconds = time.time() - cv_start

                param_name = getattr(mspec.factory(), "param_name", "param")

                test_err = np.nan
                yhat_test_full: Optional[np.ndarray] = None
                if x_te is not None and y_te is not None:
                    if rolling_test and ahead_h >= 1:
                        if rolling_test_strategy not in {"expanding", "static", "rolling"}:
                            raise ValueError("rolling_test_strategy must be 'expanding', 'static', or 'rolling'")
                        if rolling_window is not None and rolling_window <= 0:
                            raise ValueError("rolling_window must be positive when provided")

                        n_te = len(x_te)
                        step = ahead_l if ahead_l > 0 else 1
                        block_span = ahead_l if ahead_l > 0 else 0
                        target_range = range(ahead_h - 1, n_te, step)

                        if rolling_test_strategy == "expanding":
                            preds = []
                            truths = []
                            pred_positions = []
                            for tgt_offset in target_range:
                                obs_end = tgt_offset - ahead_h
                                obs_end = max(obs_end, -1)

                                x_obs = x_tr
                                y_obs = y_tr
                                if obs_end >= 0:
                                    x_obs = np.concatenate([x_tr, x_te[: obs_end + 1]])
                                    y_obs = np.concatenate([y_tr, y_te[: obs_end + 1]])

                                if rolling_window is not None and len(x_obs) > rolling_window:
                                    x_obs = x_obs[-rolling_window:]
                                    y_obs = y_obs[-rolling_window:]

                                cv_res_roll = _select_param(
                                    mspec,
                                    cv_name,
                                    x_obs,
                                    y_obs,
                                    seed=seed_train,
                                    k_folds=k_folds,
                                    buffer_l=buffer_l,
                                    walk_spec=walk_spec,
                                )

                                m_roll = mspec.factory()
                                m_roll.set_param(cv_res_roll.best_param)
                                m_roll.fit(x_obs, y_obs)
                                block_end = min(tgt_offset + block_span, n_te - 1)
                                block_x = x_te[tgt_offset : block_end + 1]
                                block_preds = m_roll.predict(block_x)
                                preds.extend(block_preds)
                                truths.extend(y_te[tgt_offset : block_end + 1])
                                pred_positions.extend(range(tgt_offset, block_end + 1))

                            if truths:
                                test_err = mse(np.asarray(truths), np.asarray(preds))
                            else:
                                test_err = np.nan

                            if return_predictions:
                                yhat_test_full = np.full(n_te, np.nan)
                                for pos, pred in zip(pred_positions, preds):
                                    yhat_test_full[pos] = pred

                        elif rolling_test_strategy == "rolling":
                            preds = []
                            truths = []
                            pred_positions = []
                            for tgt_offset in target_range:
                                obs_end = tgt_offset - ahead_h
                                obs_end = max(obs_end, -1)

                                x_hist = x_tr
                                y_hist = y_tr
                                if obs_end >= 0:
                                    x_hist = np.concatenate([x_tr, x_te[: obs_end + 1]])
                                    y_hist = np.concatenate([y_tr, y_te[: obs_end + 1]])

                                if rolling_window is not None and len(x_hist) > rolling_window:
                                    x_hist = x_hist[-rolling_window:]
                                    y_hist = y_hist[-rolling_window:]

                                cv_res_roll = _select_param(
                                    mspec,
                                    cv_name,
                                    x_hist,
                                    y_hist,
                                    seed=seed_train,
                                    k_folds=k_folds,
                                    buffer_l=buffer_l,
                                    walk_spec=walk_spec,
                                )

                                m_roll = mspec.factory()
                                m_roll.set_param(cv_res_roll.best_param)
                                m_roll.fit(x_hist, y_hist)
                                block_end = min(tgt_offset + block_span, n_te - 1)
                                block_x = x_te[tgt_offset : block_end + 1]
                                block_preds = m_roll.predict(block_x)
                                preds.extend(block_preds)
                                truths.extend(y_te[tgt_offset : block_end + 1])
                                pred_positions.extend(range(tgt_offset, block_end + 1))

                            if truths:
                                test_err = mse(np.asarray(truths), np.asarray(preds))
                            else:
                                test_err = np.nan

                            if return_predictions:
                                yhat_test_full = np.full(n_te, np.nan)
                                for pos, pred in zip(pred_positions, preds):
                                    yhat_test_full[pos] = pred

                        else:  
                            final_model_static = mspec.factory()
                            final_model_static.set_param(cv_res.best_param)
                            final_model_static.fit(x_tr, y_tr)

                            preds = final_model_static.predict(x_te[ahead_h - 1:]) if n_te >= ahead_h else np.array([])
                            truths = y_te[ahead_h - 1:] if n_te >= ahead_h else np.array([])
                            test_err = mse(truths, preds) if len(truths) else np.nan

                            if return_predictions:
                                yhat_test_full = np.full(n_te, np.nan)
                                if n_te >= ahead_h:
                                    yhat_test_full[ahead_h - 1:] = preds

                    else:
                        final_model = mspec.factory()
                        final_model.set_param(cv_res.best_param)
                        final_model.fit(x_tr, y_tr)
                        y_pred_test = final_model.predict(x_te)
                        test_err = mse(y_te, y_pred_test)
                        if return_predictions:
                            yhat_test_full = y_pred_test

                final_model = mspec.factory()
                final_model.set_param(cv_res.best_param)
                param_name = getattr(final_model, "param_name", "param")
                final_model.fit(x_tr, y_tr)
                yhat_train = final_model.predict(x_tr)

                if predictions_store is not None and rep_capture is not None:
                    rep_capture["cv_results"][cv_name] = {
                        "yhat_train": np.asarray(yhat_train),
                        "yhat_test": np.asarray(yhat_test_full) if yhat_test_full is not None else None,
                    }

                row_data = {
                    "model": mspec.name,
                    "model_key": mkey,
                    "cv_scheme": cv_name,
                    "replication": rep,
                    "seed_train": seed_train,
                    "seed_test": seed_test,
                    "param_name": param_name,
                    "selected_param": cv_res.best_param,
                    "cv_est_error": float(cv_res.best_score),
                    "test_error": float(test_err) if not np.isnan(test_err) else None,
                    "cv_time_seconds": float(cv_time_seconds),
                }
                row_data.update(dgp_info)
                rows.append(row_data)

                if save_after_each_replication and run_dir is not None:
                    metrics_path = run_dir / f"{mspec.name}_model_error.csv"
                    _append_metrics_row(metrics_path, row_data)
                    if "cv_time_seconds" in row_data:
                        time_path = run_dir / f"{mspec.name}_model_time.csv"
                        _append_metrics_row(time_path, row_data)

            if predictions_store is not None and rep_capture is not None:
                predictions_store[mkey].append(rep_capture)
                if save_after_each_replication and run_dir is not None:
                    _save_predictions_table(
                        run_dir=run_dir,
                        model_specs=model_specs,
                        predictions_store={mkey: [rep_capture]},
                        persist_format=persist_format,
                    )

    if verbose:
        total_time = time.time() - start_time
        print("-" * 60)
        print(f"Benchmark Complete. Total Time: {total_time:.2f}s")

    df = pd.DataFrame(rows)
    if "test_error" in df.columns and df["test_error"].notna().all():
        df["cv_bias"] = df["cv_est_error"] - df["test_error"]
    else:
        df["cv_bias"] = np.nan

    if return_predictions:
        return df, predictions_store
    return df




def plot_train_test_predictions(
    model_specs: Dict[str, ModelSpec],
    cv_schemes: List[str],
    *,
    dgp: Optional[DGP] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    seed: int = 123,
    k_folds: int = 5,
    buffer_l: int = 2,
    walk_spec: Optional[WalkForwardSpec] = None,
    ahead_h: int = 1,
    ahead_l: int = 0,
    rolling_test_strategy: str = "expanding",
    rolling_window: Optional[int] = None,
    save_path_png: Optional[str] = None,
    save_path_pdf: Optional[str] = None,
):

    if dgp is not None:
        sim = SyntheticSeries.generate(
            n=dgp.n,
            f_id=dgp.f_id,
            error_model=dgp.error_model,
            params=dgp.error_params,
            seed=seed,
        )

        split_idx = int(0.7 * len(sim.x))
        x_tr, y_tr = sim.x[:split_idx], sim.y[:split_idx]
        x_te, y_te = sim.x[split_idx:], sim.y[split_idx:]
    else:
        if x is None or y is None:
            raise ValueError("If DGP is not provided, x and y must be supplied.")
        if x_test is None or y_test is None:
            raise ValueError("To plot test predictions, x_test and y_test must be supplied.")
        x_tr, y_tr = np.asarray(x), np.asarray(y)
        x_te, y_te = np.asarray(x_test), np.asarray(y_test)

    n_train = len(x_tr)
    n_test = len(x_te)

    if ahead_h < 1:
        raise ValueError("ahead_h must be >= 1")
    if ahead_l < 0:
        raise ValueError("ahead_l must be >= 0")

    n_rows = len(model_specs)
    n_cols = len(cv_schemes)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 3.2 * n_rows),
        sharey=True,
    )

    axes = np.asarray(axes)
    if n_rows == 1 and n_cols == 1:
        axes = axes.reshape(1, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, mspec in enumerate(model_specs.values()):
        for j, cv_name in enumerate(cv_schemes):
            ax = axes[i, j]

            if rolling_test_strategy not in {"expanding", "static", "rolling"}:
                raise ValueError("rolling_test_strategy must be 'expanding', 'static', or 'rolling'")
            if rolling_window is not None and rolling_window <= 0:
                raise ValueError("rolling_window must be positive when provided")

            pred_x = []
            pred_y = []
            true_y = []
            used_params = []
            step = ahead_l if ahead_l > 0 else 1
            block_span = ahead_l if ahead_l > 0 else 0

            if rolling_test_strategy == "expanding":
                for tgt_offset in range(ahead_h - 1, n_test, step):
                    obs_end = tgt_offset - ahead_h
                    obs_end = max(obs_end, -1)

                    x_obs = x_tr
                    y_obs = y_tr
                    if obs_end >= 0:
                        x_obs = np.concatenate([x_tr, x_te[: obs_end + 1]])
                        y_obs = np.concatenate([y_tr, y_te[: obs_end + 1]])

                    if rolling_window is not None and len(x_obs) > rolling_window:
                        x_obs = x_obs[-rolling_window:]
                        y_obs = y_obs[-rolling_window:]

                    splits = make_splits(
                        cv_name,
                        len(x_obs),
                        seed=seed,
                        k=k_folds,
                        l=buffer_l,
                        walk_spec=walk_spec,
                    )

                    cv_res = cv_select_param(
                        model_factory=mspec.factory,
                        x=x_obs,
                        y=y_obs,
                        splits=splits,
                        grid=mspec.grid,
                        metric=mse,
                    )

                    model = mspec.factory()
                    model.set_param(cv_res.best_param)
                    model.fit(x_obs, y_obs)

                    block_end = min(tgt_offset + block_span, n_test - 1)
                    block_x = x_te[tgt_offset : block_end + 1]
                    block_preds = model.predict(block_x)
                    pred_x.extend(block_x)
                    pred_y.extend(block_preds)
                    true_y.extend(y_te[tgt_offset : block_end + 1])
                    used_params.append(cv_res.best_param)

            elif rolling_test_strategy == "rolling":
                for tgt_offset in range(ahead_h - 1, n_test, step):
                    obs_end = tgt_offset - ahead_h
                    obs_end = max(obs_end, -1)

                    x_hist = x_tr
                    y_hist = y_tr
                    if obs_end >= 0:
                        x_hist = np.concatenate([x_tr, x_te[: obs_end + 1]])
                        y_hist = np.concatenate([y_tr, y_te[: obs_end + 1]])

                    if rolling_window is not None and len(x_hist) > rolling_window:
                        x_hist = x_hist[-rolling_window:]
                        y_hist = y_hist[-rolling_window:]

                    splits = make_splits(
                        cv_name,
                        len(x_hist),
                        seed=seed,
                        k=k_folds,
                        l=buffer_l,
                        walk_spec=walk_spec,
                    )

                    cv_res = cv_select_param(
                        model_factory=mspec.factory,
                        x=x_hist,
                        y=y_hist,
                        splits=splits,
                        grid=mspec.grid,
                        metric=mse,
                    )

                    model = mspec.factory()
                    model.set_param(cv_res.best_param)
                    model.fit(x_hist, y_hist)

                    block_end = min(tgt_offset + block_span, n_test - 1)
                    block_x = x_te[tgt_offset : block_end + 1]
                    block_preds = model.predict(block_x)
                    pred_x.extend(block_x)
                    pred_y.extend(block_preds)
                    true_y.extend(y_te[tgt_offset : block_end + 1])
                    used_params.append(cv_res.best_param)

            else: 
                splits = make_splits(
                    cv_name,
                    n_train,
                    seed=seed,
                    k=k_folds,
                    l=buffer_l,
                    walk_spec=walk_spec,
                )

                cv_res = cv_select_param(
                    model_factory=mspec.factory,
                    x=x_tr,
                    y=y_tr,
                    splits=splits,
                    grid=mspec.grid,
                    metric=mse,
                )

                model = mspec.factory()
                model.set_param(cv_res.best_param)
                model.fit(x_tr, y_tr)

                target_slice = slice(ahead_h - 1, None)
                pred_x = x_te[target_slice] if n_test >= ahead_h else np.array([])
                pred_y = model.predict(pred_x) if len(pred_x) else np.array([])
                true_y = y_te[target_slice] if n_test >= ahead_h else np.array([])
                used_params.append(cv_res.best_param)

            pred_x = np.asarray(pred_x)
            pred_y = np.asarray(pred_y)
            true_y = np.asarray(true_y)

            test_err = mse(true_y, pred_y) if len(true_y) else np.nan
            pname = getattr(model, "param_name", "param")

            ax.plot(x_tr, y_tr, color="tab:blue", label="train y")
            pred_label = f"test pred (h={ahead_h}" + (f", l={ahead_l}" if ahead_l > 0 else "") + ")"
            ax.plot(pred_x, pred_y, color="tab:red", label=pred_label)
            ax.plot(pred_x, true_y, color="gray", alpha=0.4, linestyle=":", label="test y (ref)")

            ax.axvline(x_tr[-1], color="black", linestyle=":", alpha=0.6)

            subtitle = f"{pname}~{used_params[-1]} | Test MSE={test_err:.3f}" if len(used_params) else f"{pname} | Test MSE={test_err:.3f}"
            title_text = f"{cv_name}\n{subtitle}" if i == 0 else subtitle
            ax.set_title(title_text)
            if j == 0:
                ax.set_ylabel(mspec.name)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle("Train (blue) vs test predictions (red)", y=1.02)
    fig.tight_layout()
    if save_path_png:
        fig.savefig(save_path_png, bbox_inches="tight", dpi=300)
    if save_path_pdf:
        fig.savefig(save_path_pdf, bbox_inches="tight")
    plt.show()




def run_and_plot(
    model_specs: Dict[str, ModelSpec],
    cv_schemes: List[str],
    *,
    dgp: Optional[DGP] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    x_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    replications: int = 100,
    base_seed: int = 123,
    k_folds: int = 5,
    buffer_l: int = 2,
    walk_spec: Optional[WalkForwardSpec] = None,
    do_boxplots: bool = True,
    do_fit_grid: bool = True,
    do_train_test_plot: bool = True,
    ahead_h: int = 1,
    ahead_l: int = 0,
    rolling_test: bool = False,
    rolling_test_strategy: str = "expanding",
    rolling_window: Optional[int] = None,
    one_rep_seed: int = 123,
    save_boxplots_png: Optional[str] = None,
    save_boxplots_pdf: Optional[str] = None,
    save_train_test_png: Optional[str] = None,
    save_train_test_pdf: Optional[str] = None,
    save_csv_path: Optional[str] = None,
    verbose: bool = False,
    model_run_name: Optional[str] = None,
    save_after_each_replication: bool = False,
    persist_format: str = "parquet",
    fit_grid_live_fallback: bool = True,
    final_plot: bool = False,
    boxplot_yscale: Optional[str] = None,
    boxplot_metric_yscale: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:

    capture_predictions = model_run_name is not None

    skip_replications: Dict[str, Set[int]] = {}
    run_dir: Optional[Path] = None
    if model_run_name is not None:
        run_dir = Path(model_run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        for mkey, mspec in model_specs.items():
            existing_path = run_dir / f"{mspec.name}.{persist_format}"
            if existing_path.exists():
                skip_replications[mkey] = _extract_existing_replications(existing_path, persist_format)

    benchmark_output = run_cv_benchmark(
        model_specs=model_specs,
        cv_schemes=cv_schemes,
        dgp=dgp,
        x_train_fixed=x,
        y_train_fixed=y,
        x_test_fixed=x_test,
        y_test_fixed=y_test,
        replications=replications,
        base_seed=base_seed,
        k_folds=k_folds,
        buffer_l=buffer_l,
        walk_spec=walk_spec,
        rolling_test=rolling_test,
        ahead_h=ahead_h,
        ahead_l=ahead_l,
        rolling_test_strategy=rolling_test_strategy,
        rolling_window=rolling_window,
        verbose=verbose,
        return_predictions=capture_predictions,
        skip_replications=skip_replications,
        save_after_each_replication=save_after_each_replication,
        run_dir=run_dir,
        persist_format=persist_format,
    )

    auto_boxplot_path = save_boxplots_pdf
    auto_fit_path = None

    if model_run_name:
        run_name_clean = Path(model_run_name).name 
        figures_dir = Path("figures")
        
        if auto_boxplot_path is None:
            auto_boxplot_path = str(figures_dir / f"boxplot_{run_name_clean}.pdf")
            
        auto_fit_path = str(figures_dir / f"fit_{run_name_clean}.pdf")

    if capture_predictions:
        df, predictions_store = benchmark_output
    else:
        df = benchmark_output

    if save_csv_path and not df.empty:
        df.to_csv(save_csv_path, index=False)

    metrics_csv_paths: Dict[str, Path] = {}
    if model_run_name is not None:
        run_dir_path = Path(model_run_name)
        run_dir_path.mkdir(parents=True, exist_ok=True)

        if not df.empty:
            for mname in df["model"].unique():
                per_model_df = df[df["model"] == mname]
                per_path = run_dir_path / f"{mname}_model_error.csv"
                per_model_df.to_csv(per_path, index=False)
                metrics_csv_paths[mname] = per_path

                if "cv_time_seconds" in per_model_df.columns:
                    per_time_path = run_dir_path / f"{mname}_model_time.csv"
                    per_model_df.to_csv(per_time_path, index=False)
        else:
            for per_path in run_dir_path.glob("*_model_error.csv"):
                if per_path.is_file():
                    metrics_csv_paths[per_path.stem.replace("_model_error", "")] = per_path

    if df.empty:
        if verbose:
            print("[run_and_plot] No new replications were run (all skipped). Using existing saved data for plots if available.")
    else:
        if capture_predictions and not save_after_each_replication:
            _save_predictions_table(
                run_dir=Path(model_run_name),
                model_specs=model_specs,
                predictions_store=predictions_store,
                persist_format=persist_format,
            )


    if do_boxplots:
        boxplot_df: Optional[pd.DataFrame] = None

        if metrics_csv_paths:
            try:
                boxplot_df = pd.concat(
                    [pd.read_csv(p) for p in metrics_csv_paths.values()],
                    ignore_index=True,
                )
            except Exception:
                boxplot_df = None

        if (boxplot_df is None or boxplot_df.empty) and save_csv_path and Path(save_csv_path).exists():
            try:
                boxplot_df = pd.read_csv(save_csv_path)
            except Exception:
                boxplot_df = None

        if boxplot_df is None or boxplot_df.empty:
            boxplot_df = df

        if boxplot_df is not None and not boxplot_df.empty:
            metrics_to_plot = [m for m in ("test_error", "cv_est_error", "cv_bias") if m in boxplot_df.columns]

            plot_cv_boxplots_grid(
                boxplot_df,
                metrics=tuple(metrics_to_plot),
                title="CV performance boxplots",
                yscale=boxplot_yscale,
                yscale_per_metric=boxplot_metric_yscale,
                save_path_png=save_boxplots_png,
                save_path_pdf=auto_boxplot_path,
            )
        elif verbose:
            print("[run_and_plot] No data available for boxplots.")

    if do_fit_grid:
        plotted = False
        if model_run_name is not None:
            plotted = _plot_fits_from_saved(
                run_dir=Path(model_run_name),
                model_specs=model_specs,
                cv_schemes=cv_schemes,
                persist_format=persist_format,
                title_suffix="(from saved rep 0)",
                verbose=verbose,
            )

        if not plotted and fit_grid_live_fallback:
            if verbose:
                print("[run_and_plot] No saved predictions available for fit plots; falling back to live plotting.")
            plot_fits_grid_one_rep(
                model_specs=model_specs,
                cv_schemes=cv_schemes,
                dgp=dgp,
                x=x, 
                y=y,
                seed=one_rep_seed,
                k_folds=k_folds,
                buffer_l=buffer_l,
                walk_spec=walk_spec,
            )
        elif not plotted and verbose:
            print("[run_and_plot] No saved prediction data available for fit plots. Set fit_grid_live_fallback=True to refit the model.")


    if do_train_test_plot:
        has_test = (dgp is not None) or (x is not None and y is not None and x_test is not None and y_test is not None)
        if has_test:
            plot_train_test_predictions(
                model_specs=model_specs,
                cv_schemes=cv_schemes,
                dgp=dgp,
                x=x,
                y=y,
                x_test=x_test,
                y_test=y_test,
                seed=one_rep_seed,
                k_folds=k_folds,
                buffer_l=buffer_l,
                walk_spec=walk_spec,
                ahead_h=ahead_h,
                ahead_l=ahead_l,
                rolling_test_strategy=rolling_test_strategy,
                rolling_window=rolling_window,
                save_path_png=save_train_test_png,
                save_path_pdf=save_train_test_pdf,
            )
        else:
            if verbose:
                print("[run_and_plot] Skipping train/test plot (no test data or DGP provided).")

    if final_plot:
        if model_run_name is None:
            if verbose:
                print("[run_and_plot] final_plot=True requires model_run_name to load saved predictions.")
        else:
            _plot_saved_predictions_grid(
                run_dir=Path(model_run_name),
                model_specs=model_specs,
                cv_schemes=cv_schemes,
                persist_format=persist_format,
                verbose=verbose,
                save_path_pdf=auto_fit_path
            )

    return df





def _select_first_rep_tag(columns: Iterable[str]) -> Optional[str]:
    rep_pattern = re.compile(r"(rep(\d+)_seed\d+)_y$")
    candidates: List[Tuple[int, str]] = []
    for col in columns:
        match = rep_pattern.match(str(col))
        if match:
            rep_id = int(match.group(2))
            candidates.append((rep_id, match.group(1)))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[0][1]


def _nan_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    diff = y_true[mask] - y_pred[mask]
    return float(np.mean(diff * diff))


def _list_rep_tags(columns: Iterable[str]) -> List[Tuple[str, int, int]]:
    rep_pattern = re.compile(r"rep(\d+)_seed(\d+)_y$")
    out: List[Tuple[str, int, int]] = []
    for col in columns:
        m = rep_pattern.match(str(col))
        if m:
            rep_id = int(m.group(1))
            seed = int(m.group(2))
            out.append((f"rep{rep_id}_seed{seed}", rep_id, seed))
    return out


def _build_metrics_from_saved_predictions(
    *,
    run_dir: Path,
    model_specs: Dict[str, ModelSpec],
    cv_schemes: List[str],
    persist_format: str = "parquet",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for mkey, mspec in model_specs.items():
        path = run_dir / f"{mspec.name}.{persist_format}"
        if not path.exists():
            continue
        try:
            table = _load_existing_table(path, persist_format)
        except Exception:
            continue

        ahead_l=ahead_l,
        if "split" in table.columns:
            split = table["split"].to_numpy()
            test_mask = split == "test"
        else:
            split = None
            test_mask = None

        rep_tags = _list_rep_tags(table.columns)
        for rep_tag, rep_id, seed in rep_tags:
            y_col = f"{rep_tag}_y"
            if y_col not in table.columns:
                continue
            y_true = table[y_col].to_numpy()

            for cv_name in cv_schemes:
                pred_col = f"{rep_tag}_{cv_name}_yhat"
                if pred_col not in table.columns:
                    continue
                preds = table[pred_col].to_numpy()

                if test_mask is not None:
                    mse_test = _nan_mse(y_true[test_mask], preds[test_mask])
                else:
                    mse_test = _nan_mse(y_true, preds)

                rows.append(
                    {
                        "model": mspec.name,
                        "model_key": mkey,
                        "cv_scheme": cv_name,
                        "replication": rep_id,
                        "seed_train": seed,
                        "seed_test": None,
                        "param_name": None,
                        "selected_param": None,
                        "cv_est_error": np.nan,
                        "test_error": mse_test,
                        "cv_bias": np.nan,
                    }
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _overlay_test_error(recomputed: pd.DataFrame, csv_df: pd.DataFrame) -> pd.DataFrame:
    if recomputed.empty:
        return csv_df

    key_cols = ["model", "cv_scheme", "replication"]
    if "replication" not in csv_df.columns:
        return csv_df

    merged = pd.merge(csv_df, recomputed[key_cols + ["test_error"]], on=key_cols, how="left", suffixes=("", "_recomp"))

    merged["test_error"] = merged["test_error_recomp"].combine_first(merged["test_error"])
    merged.drop(columns=[c for c in merged.columns if c.endswith("_recomp")], inplace=True)

    if "cv_est_error" in merged.columns:
        merged["cv_bias"] = merged["cv_est_error"] - merged["test_error"]

    return merged


def _plot_fits_from_saved(
    *,
    run_dir: Path,
    model_specs: Dict[str, ModelSpec],
    cv_schemes: List[str],
    persist_format: str = "parquet",
    title_suffix: str = "",
    verbose: bool = False,
) -> bool:

    model_list = list(model_specs.values())
    per_model: Dict[str, Dict[str, Any]] = {}

    for mspec in model_list:
        path = run_dir / f"{mspec.name}.{persist_format}"
        if not path.exists():
            if verbose:
                print(f"[_plot_fits_from_saved] No saved data file found for model '{mspec.name}' at {path}")
            continue
        try:
            table = _load_existing_table(path, persist_format)
        except Exception as e:
            if verbose:
                print(f"[_plot_fits_from_saved] Could not load saved data for model '{mspec.name}': {e}")
            continue

        rep_tag = _select_first_rep_tag(table.columns)
        if rep_tag is None:
            if verbose:
                print(f"[_plot_fits_from_saved] No replication data found in saved file for model '{mspec.name}'")
            continue

        y_col = f"{rep_tag}_y"
        if y_col not in table.columns:
            continue

        per_model[mspec.name] = {
            "x": table["x"].to_numpy(),
            "y": table[y_col].to_numpy(),
            "f_true": table["f_true"].to_numpy() if "f_true" in table.columns else None,
            "split": table["split"].to_numpy() if "split" in table.columns else None,
            "rep_tag": rep_tag,
            "preds": {},
        }

        for cv_name in cv_schemes:
            pred_col = f"{rep_tag}_{cv_name}_yhat"
            if pred_col in table.columns:
                per_model[mspec.name]["preds"][cv_name] = table[pred_col].to_numpy()

    if not per_model:
        if verbose:
            print("[_plot_fits_from_saved] No saved prediction data available for any model. Skipping fit plot.")
        return False

    n_rows = len(model_list)
    n_cols = len(cv_schemes)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3.0 * n_rows),
        sharex=True,
        sharey=True,
    )

    axes = np.asarray(axes)
    if n_rows == 1 and n_cols == 1:
        axes = axes.reshape(1, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    plotted_any = False

    for i, mspec in enumerate(model_list):
        info = per_model.get(mspec.name)
        split = info.get("split") if info else None
        boundary_idx: Optional[int] = None
        if split is not None:
            test_positions = np.where(split == "test")[0]
            if len(test_positions):
                boundary_idx = int(test_positions[0])

        for j, cv_name in enumerate(cv_schemes):
            ax = axes[i, j]

            if info is None:
                ax.text(0.5, 0.5, "no saved data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
                if i == 0:
                    ax.set_title(cv_name)
                if j == 0:
                    ax.set_ylabel(mspec.name)
                continue

            yhat = info["preds"].get(cv_name)

            ax.plot(info["x"], info["y"], alpha=0.5, label="observed y", color="gray")
            if info.get("f_true") is not None:
                ax.plot(info["x"], info["f_true"], linewidth=1.6, label="true f", color="black", linestyle="--")

            if yhat is not None:
                ax.plot(info["x"], yhat, linewidth=2, label="fitted", color="tab:blue")
                plotted_any = True
            else:
                ax.text(0.5, 0.2, "no saved yhat", ha="center", va="center", transform=ax.transAxes, fontsize=8)

            if boundary_idx is not None and boundary_idx < len(info["x"]):
                ax.axvline(info["x"][boundary_idx], color="black", linestyle=":", alpha=0.6)

            sel_param, test_err = _lookup_metrics(
                run_dir=run_dir,
                model_name=mspec.name,
                cv_name=cv_name,
                rep_tag=info.get("rep_tag"),
            )
            sel_param_fmt = _format_two_dec(sel_param)
            test_err_fmt = _format_two_dec(test_err)

            subtitle_parts: List[str] = []
            if sel_param_fmt is not None:
                subtitle_parts.append(f"param={sel_param_fmt}")
            if test_err_fmt is not None:
                subtitle_parts.append(f"test={test_err_fmt}")
            subtitle = " | ".join(subtitle_parts)

            if i == 0:
                title_text = f"{cv_name}\n{subtitle}" if subtitle else cv_name
            else:
                title_text = subtitle if subtitle else ""

            ax.set_title(title_text)
            if j == 0:
                ax.set_ylabel(mspec.name)

    if not plotted_any:
        plt.close(fig)
        return False

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    suffix = f" {title_suffix}" if title_suffix else ""
    fig.suptitle(f"One-rep fitted curves by model (saved){suffix}", y=1.02)
    fig.tight_layout()
    plt.show()
    return True




def _format_two_dec(val: Any) -> Optional[str]:
    if val is None:
        return None
    try:
        if isinstance(val, (float, np.floating)) and np.isnan(val):
            return None
    except Exception:
        return None
    if isinstance(val, (int, float, np.floating)):
        return f"{float(val):.2f}"
    return str(val)


def _lookup_metrics(
    *,
    run_dir: Path,
    model_name: str,
    cv_name: str,
    rep_tag: str,
) -> Tuple[Optional[Any], Optional[float]]:
    metrics_path = run_dir / f"{model_name}_model_error.csv"
    if not metrics_path.exists():
        return None, None
    try:
        mdf = pd.read_csv(metrics_path)
    except Exception:
        return None, None

    rep_match = re.match(r"rep(\d+)_seed\d+", rep_tag)
    if rep_match is None or "replication" not in mdf.columns:
        return None, None
    rep_id = int(rep_match.group(1))

    row = mdf[(mdf.get("cv_scheme") == cv_name) & (mdf.get("replication") == rep_id)]
    if row.empty:
        return None, None

    sel_param = row.iloc[0].get("selected_param", None)
    test_error = row.iloc[0].get("test_error", None)
    try:
        test_error = float(test_error)
    except Exception:
        pass
    return sel_param, test_error


def _plot_saved_predictions_grid(
    *,
    run_dir: Path,
    model_specs: Dict[str, ModelSpec],
    cv_schemes: List[str],
    persist_format: str = "parquet",
    verbose: bool = False,
    save_path_pdf: Optional[str] = None
) -> bool:
    model_list = list(model_specs.values())
    model_data: List[Dict[str, Any]] = []

    metrics_cache: Dict[Tuple[str, str, Optional[int]], Tuple[Optional[Any], Optional[float]]] = {}
    for mspec in model_list:
        mpath = run_dir / f"{mspec.name}_model_error.csv"
        if not mpath.exists():
            continue
        try:
            mdf = pd.read_csv(mpath)
        except Exception:
            continue

        has_rep = "replication" in mdf.columns
        for _, row in mdf.iterrows():
            rep_id = int(row["replication"]) if has_rep else None
            key = (mspec.name, str(row.get("cv_scheme", "")), rep_id)
            sel_param = row.get("selected_param", None)
            test_err = row.get("test_error", None)
            metrics_cache[key] = (sel_param, test_err)
            generic_key = (mspec.name, str(row.get("cv_scheme", "")), None)
            if generic_key not in metrics_cache:
                metrics_cache[generic_key] = (sel_param, test_err)

    for mspec in model_list:
        path = run_dir / f"{mspec.name}.{persist_format}"
        if not path.exists():
            model_data.append({"name": mspec.name, "status": f"No saved file at {path}"})
            if verbose:
                print(f"[_plot_saved_predictions_grid] Missing saved table for {mspec.name}: {path}")
            continue

        try:
            table = _load_existing_table(path, persist_format)
        except Exception as e:
            model_data.append({"name": mspec.name, "status": f"Could not load saved data: {e}"})
            if verbose:
                print(f"[_plot_saved_predictions_grid] Load failed for {mspec.name}: {e}")
            continue

        rep_tag = _select_first_rep_tag(table.columns)
        if rep_tag is None:
            model_data.append({"name": mspec.name, "status": "No replication columns found"})
            if verbose:
                print(f"[_plot_saved_predictions_grid] No replication tag for {mspec.name}")
            continue

        if "x" not in table.columns:
            model_data.append({"name": mspec.name, "status": "No x column in saved table"})
            if verbose:
                print(f"[_plot_saved_predictions_grid] Missing x column for {mspec.name}")
            continue

        y_col = f"{rep_tag}_y"
        if y_col not in table.columns:
            model_data.append({"name": mspec.name, "status": f"Missing {y_col} column"})
            if verbose:
                print(f"[_plot_saved_predictions_grid] Missing {y_col} for {mspec.name}")
            continue

        split = table["split"].to_numpy() if "split" in table.columns else None
        boundary_idx: Optional[int] = None
        if split is not None:
            test_positions = np.where(split == "test")[0]
            if len(test_positions):
                boundary_idx = int(test_positions[0])

        model_data.append(
            {
                "name": mspec.name,
                "rep_tag": rep_tag,
                "table": table,
                "x": table["x"].to_numpy(),
                "y_true": table[y_col].to_numpy(),
                "split": split,
                "boundary_idx": boundary_idx,
            }
        )

    if not model_data:
        if verbose:
            print("[_plot_saved_predictions_grid] No model data available to plot.")
        return False

    n_rows = len(model_list)
    n_cols = len(cv_schemes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.4 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    plotted = False
    for i, mspec in enumerate(model_list):
        info = model_data[i]
        for j, cv_name in enumerate(cv_schemes):
            ax = axes[i, j]

            if "status" in info:
                ax.text(0.5, 0.5, info["status"], transform=ax.transAxes, ha="center", va="center", fontsize=9)
                if i == 0:
                    ax.set_title(cv_name)
                if j == 0:
                    ax.set_ylabel(mspec.name)
                continue

            table = info["table"]
            rep_tag = info["rep_tag"]
            x_vals = info["x"]
            y_true = info["y_true"]
            split = info["split"]
            boundary_idx = info["boundary_idx"]

            pred_col = f"{rep_tag}_{cv_name}_yhat"
            if pred_col not in table.columns:
                ax.text(0.5, 0.5, f"missing {pred_col}", transform=ax.transAxes, ha="center", va="center", fontsize=9)
                if i == 0:
                    ax.set_title(cv_name)
                if j == 0:
                    ax.set_ylabel(mspec.name)
                continue

            preds = table[pred_col].to_numpy()
            ax.plot(x_vals, y_true, color="gray", alpha=0.6, label="true y")

            if split is not None:
                train_mask = split == "train"
                test_mask = split == "test"
                if train_mask.any():
                    ax.plot(x_vals[train_mask], preds[train_mask], color="tab:blue", linewidth=2, label="train pred")
                if test_mask.any():
                    ax.plot(x_vals[test_mask], preds[test_mask], color="tab:red", linewidth=2, label="test pred")
            else:
                ax.plot(x_vals, preds, color="tab:blue", linewidth=2, label="pred")

            if boundary_idx is not None and boundary_idx < len(x_vals):
                ax.axvline(x_vals[boundary_idx], color="black", linestyle=":", alpha=0.6)

            sel_param, test_err = _lookup_metrics(run_dir=run_dir, model_name=mspec.name, cv_name=cv_name, rep_tag=rep_tag)
            if sel_param is None and test_err is None:
                rep_match = re.match(r"rep(\d+)_seed", rep_tag)
                rep_id = int(rep_match.group(1)) if rep_match else None
                sel_param, test_err = metrics_cache.get((mspec.name, cv_name, rep_id), (None, None))
                if sel_param is None and test_err is None:
                    sel_param, test_err = metrics_cache.get((mspec.name, cv_name, None), (None, None))
            sel_param_fmt = _format_two_dec(sel_param)
            test_err_fmt = _format_two_dec(test_err)
            
            subtitle_parts: List[str] = []
            if sel_param_fmt is not None:
                subtitle_parts.append(f"param={sel_param_fmt}")
            if test_err_fmt is not None:
                subtitle_parts.append(f"test={test_err_fmt}")
            subtitle = " | ".join(subtitle_parts)

            if i == 0:
                ax.set_title(cv_name, fontsize=15, fontweight='bold', pad=30)
                
                if subtitle:
                    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, 
                            ha='center', va='bottom', fontsize=15, fontweight='normal')
            else:
                if subtitle:
                    ax.set_title(subtitle, fontsize=15, fontweight='normal')

            if j == 0:
                ax.set_ylabel(mspec.name, fontsize=15, fontweight='bold')
            if i == 0 and j == 0:
                ax.legend(loc="upper right")

            plotted = True

    if not axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend(loc="upper right")

    if not plotted:
        plt.close(fig)
        if verbose:
            print("[_plot_saved_predictions_grid] Nothing plotted (no predictions found).")
        return False

    fig.suptitle("Predictions vs true by model (rows) and CV scheme (cols)", y=1.03, fontsize=18, fontweight='bold')
    fig.tight_layout()

    if save_path_pdf:
        Path(save_path_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path_pdf, bbox_inches="tight", dpi=300)
        if verbose:
            print(f"Saved fit grid to: {save_path_pdf}")

    plt.show()
    return True




def _write_table(df: pd.DataFrame, path: Path, persist_format: str) -> None:
    if persist_format == "parquet":
        df.to_parquet(path, index=False)
    elif persist_format == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported persist_format '{persist_format}'. Use 'parquet' or 'csv'.")


def _append_metrics_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def _load_existing_table(path: Path, persist_format: str) -> pd.DataFrame:
    if persist_format == "parquet":
        return pd.read_parquet(path)
    if persist_format == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported persist_format '{persist_format}'. Use 'parquet' or 'csv'.")


def _extract_existing_replications(path: Path, persist_format: str = "parquet") -> Set[int]:

    if not path.exists():
        return set()

    rep_pattern = re.compile(r"rep(\d+)_seed")
    try:
        df = _load_existing_table(path, persist_format)
    except Exception:
        return set()

    reps: Set[int] = set()
    for col in df.columns:
        match = rep_pattern.search(str(col))
        if match:
            reps.add(int(match.group(1)))
    return reps


def _save_predictions_table(
    *,
    run_dir: Path,
    model_specs: Dict[str, ModelSpec],
    predictions_store: Optional[Dict[str, List[Dict[str, Any]]]],
    persist_format: str = "parquet",
):

    if predictions_store is None:
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    for mkey, rep_list in predictions_store.items():
        if not rep_list:
            continue

        model_name = model_specs[mkey].name
        path = run_dir / f"{model_name}.{persist_format}"
        base_df: Optional[pd.DataFrame]

        if path.exists():
            base_df = _load_existing_table(path, persist_format)
            base_split = base_df["split"].to_numpy()
            base_x = base_df["x"].to_numpy()
        else:
            base_df = None
            base_split = None
            base_x = None

        for rec in rep_list:
            x_tr = np.asarray(rec["x_train"])
            y_tr = np.asarray(rec["y_train"])
            f_tr = rec.get("f_train")
            x_te = np.asarray(rec["x_test"]) if rec.get("x_test") is not None else None
            y_te = np.asarray(rec["y_test"]) if rec.get("y_test") is not None else None
            f_te = np.asarray(rec["f_test"]) if rec.get("f_test") is not None else None

            if x_te is not None:
                full_x = np.concatenate([x_tr, x_te])
                split_labels = np.array(["train"] * len(x_tr) + ["test"] * len(x_te))
            else:
                full_x = x_tr
                split_labels = np.array(["train"] * len(x_tr))

            if base_df is None:
                base_df = pd.DataFrame({"split": split_labels, "x": full_x})
                base_split = split_labels
                base_x = full_x
                if f_tr is not None:
                    if f_te is not None:
                        base_df["f_true"] = np.concatenate([f_tr, f_te])
                    else:
                        base_df["f_true"] = np.asarray(f_tr)
            else:
                if len(base_df) != len(full_x) or not np.array_equal(base_split, split_labels):
                    raise ValueError("Existing parquet shape/split mismatch with current run.")
                if not np.allclose(base_x, full_x):
                    raise ValueError("Existing parquet x grid differs from current run; refusing to merge.")
                if "f_true" not in base_df.columns and f_tr is not None:
                    if f_te is not None:
                        base_df["f_true"] = np.concatenate([f_tr, f_te])
                    else:
                        base_df["f_true"] = np.asarray(f_tr)

            rep_tag = f"rep{rec['replication']}_seed{rec['seed_train']}"

            if x_te is not None and y_te is not None:
                full_y = np.concatenate([y_tr, y_te])
            elif x_te is not None:
                full_y = np.concatenate([y_tr, np.full(len(x_te), np.nan)])
            else:
                full_y = y_tr

            col_y = f"{rep_tag}_y"
            if col_y not in base_df.columns:
                base_df[col_y] = full_y

            col_seed = f"{rep_tag}_seed_value"
            if col_seed not in base_df.columns:
                base_df[col_seed] = np.full(len(full_x), rec["seed_train"])

            for cv_name, cv_data in rec.get("cv_results", {}).items():
                yhat_tr = cv_data.get("yhat_train")
                yhat_te = cv_data.get("yhat_test")

                if yhat_tr is None:
                    continue

                if x_te is not None:
                    if yhat_te is None:
                        yhat_te = np.full(len(x_te), np.nan)
                    full_pred = np.concatenate([yhat_tr, yhat_te])
                else:
                    full_pred = yhat_tr

                col_pred = f"{rep_tag}_{cv_name}_yhat"
                if col_pred in base_df.columns:
                    continue
                base_df[col_pred] = full_pred

        if base_df is not None:
            _write_table(base_df, path, persist_format)


def _load_existing_table(path: Path, persist_format: str) -> pd.DataFrame:
    if persist_format == "parquet":
        return pd.read_parquet(path)
    if persist_format == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported persist_format '{persist_format}'. Use 'parquet' or 'csv'.")