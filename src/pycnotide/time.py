"""UTC-safe time normalization."""

from __future__ import annotations

import numpy as np


def as_datetime64_ns(value: str | np.datetime64) -> np.datetime64:
    """Parse an explicitly UTC ISO string without NumPy's timezone warning."""

    if isinstance(value, str):
        value = value.removesuffix("Z")
    return np.datetime64(value, "ns")


def normalize_time(
    time: np.ndarray,
    reference_time: str | np.datetime64 | None = None,
) -> tuple[np.ndarray, np.datetime64]:
    """Return seconds from an explicit nanosecond-resolution UTC-like epoch."""

    values = np.asarray(time)
    if np.issubdtype(values.dtype, np.datetime64):
        dt = values.astype("datetime64[ns]")
        finite = ~np.isnat(dt)
        if not np.any(finite):
            raise ValueError("time has no finite timestamps")
        ref = (
            as_datetime64_ns(reference_time)
            if reference_time is not None
            else np.min(dt[finite])
        )
        seconds = (dt - ref) / np.timedelta64(1, "s")
        return np.asarray(seconds, dtype=float), ref
    seconds = np.asarray(values, dtype=float)
    if reference_time is None:
        ref = np.datetime64("1970-01-01T00:00:00", "ns")
    else:
        ref = np.datetime64(reference_time, "ns")
    return seconds, ref
