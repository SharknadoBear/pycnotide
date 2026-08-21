"""Weighted harmonic design-matrix construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DesignMatrix:
    values: np.ndarray
    background_terms: tuple[str, ...]
    harmonic_start: int
    theta: np.ndarray


def _scaled_column(values: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return None
    centered = values - np.nanmean(values[finite])
    scale = np.nanstd(centered[finite])
    if not np.isfinite(scale) or scale <= 0:
        return None
    return centered / scale


def build_design_matrix(
    time_seconds: np.ndarray,
    frequencies: np.ndarray,
    *,
    phase_offset: np.ndarray | None = None,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    background: str = "linear_space_time",
) -> DesignMatrix:
    """Build columns for nuisance background and complex harmonic coefficients."""

    t = np.asarray(time_seconds, dtype=float).reshape(-1)
    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
    if phase_offset is None:
        phase = np.zeros((frequencies.size, t.size), dtype=float)
    else:
        phase = np.asarray(phase_offset, dtype=float)
        if phase.shape != (frequencies.size, t.size):
            raise ValueError("phase_offset must have shape (constituent, observation)")
    theta = frequencies[:, None] * t[None, :] + phase
    columns: list[np.ndarray] = [np.ones(t.size)]
    names = ["intercept"]
    mode = str(background).lower()
    if mode not in {"constant", "linear_time", "linear_space", "linear_space_time"}:
        raise ValueError(f"Unsupported background {background!r}")
    if mode in {"linear_time", "linear_space_time"}:
        col = _scaled_column(t)
        if col is not None:
            columns.append(col)
            names.append("time")
    if mode in {"linear_space", "linear_space_time"}:
        for label, raw in (("x", x), ("y", y)):
            if raw is not None:
                col = _scaled_column(np.asarray(raw, dtype=float).reshape(-1))
                if col is not None:
                    columns.append(col)
                    names.append(label)
    harmonic_start = len(columns)
    for row in theta:
        columns.extend((np.cos(row), -np.sin(row)))
    return DesignMatrix(np.column_stack(columns), tuple(names), harmonic_start, theta)


def observation_weights(
    mode: str | np.ndarray,
    n: int,
    group: np.ndarray | None = None,
) -> np.ndarray:
    """Return normalized nonnegative least-squares weights."""

    if not isinstance(mode, str):
        weights = np.asarray(mode, dtype=float).reshape(-1)
        if weights.size != n:
            raise ValueError("custom weights must match the observation count")
    elif mode in {"sample", "profile"}:
        weights = np.ones(n, dtype=float)
    elif mode == "group":
        if group is None:
            raise ValueError("group weights require group labels")
        labels = np.asarray(group).reshape(-1)
        if labels.size != n:
            raise ValueError("group labels must match the observation count")
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse]
    else:
        raise ValueError(f"Unsupported weight mode {mode!r}")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weights must be finite, nonnegative, and contain a positive value")
    return weights * (n / np.sum(weights))

