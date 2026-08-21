"""Complete-block bootstrap utilities."""

from __future__ import annotations

import numpy as np


def block_bootstrap_indices(
    time_seconds: np.ndarray,
    *,
    block_seconds: float = 48.0 * 3600.0,
    n_resamples: int = 1000,
    seed: int = 20191003,
) -> np.ndarray:
    """Return deterministic complete-time-block resampling indices."""

    time = np.asarray(time_seconds, dtype=float).reshape(-1)
    if block_seconds <= 0 or n_resamples <= 0:
        raise ValueError("block_seconds and n_resamples must be positive")
    finite = np.isfinite(time)
    if not np.all(finite):
        raise ValueError("time_seconds must be finite")
    block = np.floor((time - np.min(time)) / block_seconds).astype(int)
    labels = np.unique(block)
    members = [np.flatnonzero(block == label) for label in labels]
    rng = np.random.default_rng(seed)
    output = np.empty((n_resamples, time.size), dtype=int)
    for i in range(n_resamples):
        pieces: list[np.ndarray] = []
        while sum(piece.size for piece in pieces) < time.size:
            pieces.append(members[int(rng.integers(len(members)))])
        output[i] = np.concatenate(pieces)[: time.size]
    return output

