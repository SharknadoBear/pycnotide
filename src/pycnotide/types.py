"""Immutable result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HarmonicResult:
    constituents: tuple[str, ...]
    frequency_rad_s: np.ndarray
    depth: np.ndarray
    reference_time: np.datetime64
    phase_convention: str
    complex_density: np.ndarray
    complex_displacement: np.ndarray
    amplitude_density: np.ndarray
    amplitude_displacement: np.ndarray
    phase_lag: np.ndarray
    standard_error_density: np.ndarray
    harmonic_variance: np.ndarray
    background_density: np.ndarray
    density_gradient: np.ndarray
    background_coefficients: np.ndarray
    background_terms: tuple[str, ...]
    valid_count: np.ndarray
    variance_explained: np.ndarray
    phase_coverage: np.ndarray
    design_rank: np.ndarray
    design_condition: np.ndarray
    resolved_constituent: np.ndarray
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_xarray(self):
        """Convert to an xarray Dataset without making xarray a core dependency."""

        try:
            import xarray as xr
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install xarray to use HarmonicResult.to_xarray()") from exc
        coords = {"constituent": list(self.constituents), "depth": self.depth}
        return xr.Dataset(
            data_vars={
                "complex_density": (("constituent", "depth"), self.complex_density),
                "complex_displacement": (
                    ("constituent", "depth"),
                    self.complex_displacement,
                ),
                "amplitude_density": (("constituent", "depth"), self.amplitude_density),
                "amplitude_displacement": (
                    ("constituent", "depth"),
                    self.amplitude_displacement,
                ),
                "phase_lag": (("constituent", "depth"), self.phase_lag),
                "standard_error_density": (
                    ("constituent", "depth"),
                    self.standard_error_density,
                ),
                "variance_explained": (("depth",), self.variance_explained),
                "valid_count": (("depth",), self.valid_count),
                "density_gradient": (("depth",), self.density_gradient),
            },
            coords=coords,
            attrs={
                "reference_time": str(self.reference_time),
                "phase_convention": self.phase_convention,
                **{str(k): str(v) for k, v in self.metadata.items()},
            },
        )


@dataclass(frozen=True)
class Prediction:
    time: np.ndarray
    depth: np.ndarray
    xi: np.ndarray
    xi_by_constituent: np.ndarray
    w_iso: np.ndarray
    w_iso_by_constituent: np.ndarray
    rho: np.ndarray
    rho_anomaly: np.ndarray
    rho_anomaly_by_constituent: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseFitResult:
    constituent: str
    direction_deg: float
    phase_speed_m_s: float
    objective: float
    success: bool
    n_observations: int
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

