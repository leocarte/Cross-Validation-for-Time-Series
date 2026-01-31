from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple, Optional
import numpy as np


Split = Tuple[np.ndarray, np.ndarray]



def naive_kfold_indices(
    n: int,
    k: int = 5,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    if n <= 0:
        raise ValueError("n must be positive.")
    if k <= 1:
        raise ValueError("k must be >= 2.")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return [np.sort(f) for f in folds]


def naive_kfold_splits(
    n: int,
    k: int = 5,
    seed: Optional[int] = None,
) -> Iterator[Split]:
    folds = naive_kfold_indices(n, k=k, seed=seed)
    all_idx = np.arange(n)
    for val_idx in folds:
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        train_idx = all_idx[mask]
        yield train_idx, val_idx



def block_kfold_indices(n: int, k: int = 5) -> List[np.ndarray]:
    if n <= 0:
        raise ValueError("n must be positive.")
    if k <= 1:
        raise ValueError("k must be >= 2.")

    idx = np.arange(n)
    blocks = np.array_split(idx, k)
    return [b for b in blocks]


def block_kfold_splits(n: int, k: int = 5) -> Iterator[Split]:
    blocks = block_kfold_indices(n, k=k)
    all_idx = np.arange(n)
    for val_idx in blocks:
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        train_idx = all_idx[mask]
        yield train_idx, val_idx



def leave_2l_plus_1_out_splits(
    n: int,
    l: int = 2,
) -> Iterator[Split]:

    if l < 0:
        raise ValueError("l must be >= 0.")

    all_idx = np.arange(n)

    for i in range(n):
        val_idx = np.array([i], dtype=int)

        b_start = max(0, i - l)
        b_end = min(n - 1, i + l)
        
        mask = np.ones(n, dtype=bool)
        
        mask[b_start : b_end + 1] = False
        
        train_idx = all_idx[mask]

        yield train_idx, val_idx




def block_with_buffer_splits(
    n: int,
    k: int = 10,
    l: int = 2,
) -> Iterator[Split]:

    if n <= 0:
        raise ValueError("n must be positive.")
    if k <= 1:
        raise ValueError("k must be >= 2.")
    if l < 0:
        raise ValueError("l must be >= 0.")

    all_idx = np.arange(n)
    blocks = np.array_split(all_idx, k)

    for val_idx in blocks:
        if len(val_idx) == 0:
            continue
        b_start = max(0, int(val_idx[0]) - l)
        b_end = min(n - 1, int(val_idx[-1]) + l)

        mask = np.ones(n, dtype=bool)
        mask[b_start : b_end + 1] = False

        train_idx = all_idx[mask]
        yield train_idx, val_idx




@dataclass(frozen=True)
class WalkForwardSpec:

    initial_train_size: int
    val_size: int
    step: int = 1
    max_splits: Optional[int] = None


def walk_forward_splits(n: int, spec: WalkForwardSpec) -> Iterator[Split]:
    if n <= 0:
        raise ValueError("n must be positive.")
    if spec.initial_train_size <= 0 or spec.val_size <= 0 or spec.step <= 0:
        raise ValueError("initial_train_size, val_size, step must be > 0.")

    split_count = 0
    train_end = spec.initial_train_size

    while True:
        val_start = train_end
        val_end = val_start + spec.val_size

        if val_end > n:
            break

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)

        yield train_idx, val_idx

        split_count += 1
        if spec.max_splits is not None and split_count >= spec.max_splits:
            break

        train_end += spec.step




def assert_disjoint(train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    inter = np.intersect1d(train_idx, val_idx)
    if len(inter) != 0:
        raise AssertionError("Train and validation sets overlap.")


def assert_buffer_excluded(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    l: int,
    n: int
) -> None:
    if l <= 0:
        return
    v_start = int(val_idx[0])
    v_end = int(val_idx[-1])
    b_start = max(0, v_start - l)
    b_end = min(n - 1, v_end + l)

    forbidden = set(range(b_start, b_end + 1))
    train_set = set(map(int, train_idx.tolist()))
    if forbidden & train_set:
        raise AssertionError("Leave 2l+1 out CV: buffer region leaked into training.")
