from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

from utils.signals import generate_signal, SIGNALS
from utils.errors import get_error_model, ERROR_MODELS
from utils.acf import empirical_acf, plot_acf


@dataclass
class SyntheticSeries:

    x: np.ndarray
    f: np.ndarray
    eps: np.ndarray
    y: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


    @classmethod
    def generate(
        cls,
        *,
        n: int,
        f_id: str,
        error_model: str,
        params: Optional[Dict[str, Any]] = None,
        x_kind: str = "time",
        standardize_signal: bool = True,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> "SyntheticSeries":

        if n <= 0:
            raise ValueError("n must be positive.")
        if f_id not in SIGNALS:
            raise KeyError(f"Unknown f_id '{f_id}'. Available: {list(SIGNALS.keys())}")
        if error_model not in ERROR_MODELS:
            raise KeyError(f"Unknown error_model '{error_model}'. Available: {list(ERROR_MODELS.keys())}")

        params = dict(params or {})

        if rng is None:
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

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
            "seed": seed,
        }

        return cls(x=x, f=f, eps=eps, y=y, meta=meta)



    def acf(self, which: str = "y", nlags: int = 40, demean: bool = True) -> np.ndarray:

        series = self._get_component(which)
        return empirical_acf(series, nlags=nlags, demean=demean)

    def _get_component(self, which: str) -> np.ndarray:
        if which == "y":
            return self.y
        if which == "eps":
            return self.eps
        if which == "f":
            return self.f
        raise ValueError("which must be one of: 'y', 'eps', 'f'.")



    def plot_series(
        self,
        *,
        show_f: bool = True,
        show_eps: bool = False,
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.x, self.y, label="y")

        if show_f:
            plt.plot(self.x, self.f, label="f(x)")

        if show_eps:
            plt.plot(self.x, self.eps, label="eps", alpha=0.7)

        default_title = f"Synthetic series (f_id={self.meta.get('f_id')}, error={self.meta.get('error_model')})"
        plt.title(title or default_title)
        plt.xlabel("t" if self.meta.get("x_kind") == "time" else "x")
        plt.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_acf(
        self,
        *,
        which: str = "y",
        nlags: int = 40,
        demean: bool = True,
        show_confidence: bool = True,
        alpha: float = 0.05,
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:

        series = self._get_component(which)

        default_title = f"ACF of {which} (f_id={self.meta.get('f_id')}, error={self.meta.get('error_model')})"

        plot_acf(
            series,
            nlags=nlags,
            title=title or default_title,
            demean=demean,
            show_confidence=show_confidence,
            alpha=alpha,
            show=show,
            save_path=save_path,
        )

    def plot_acf_y(self, **kwargs) -> None:
        self.plot_acf(which="y", **kwargs)

    def plot_acf_eps(self, **kwargs) -> None:
        self.plot_acf(which="eps", **kwargs)

    def plot_acf_f(self, **kwargs) -> None:
        self.plot_acf(which="f", **kwargs)
