from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from utils.signals import SIGNALS, generate_signal

def plot_all_signals(
    n: int = 300,
    x_start: float = 0.0,
    x_end: float = 100.0,
    standardize: bool = True,
    show: bool = True,
    save_path: str | None = None,
    size: tuple = (10, 6),
):

    x = np.linspace(x_start, x_end, n)

    plt.figure(figsize=size)
    for name in SIGNALS.keys():
        y = generate_signal(x, name, standardize=standardize)
        plt.plot(x, y, label=name)

    plt.title(f"True signal library (standardize={standardize})")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_each_signal_separately(
    n: int = 300,
    x_start: float = 0.0,
    x_end: float = 100.0,
    standardize: bool = True,
    show: bool = True,
    save_dir: str | None = None,
    size: tuple = (10, 6),
):

    x = np.linspace(x_start, x_end, n)

    for name in SIGNALS.keys():
        y = generate_signal(x, name, standardize=standardize)

        plt.figure(figsize=size)
        plt.plot(x, y)
        plt.title(f"{name} (standardize={standardize})")
        plt.xlabel("x")
        plt.ylabel("f(x)")

        if save_dir is not None:
            plt.savefig(f"{save_dir}/{name}_std_{standardize}.png", bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()