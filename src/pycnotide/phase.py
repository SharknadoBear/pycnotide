"""Spatial phase models and current-aware straight-ray eikonal integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution, minimize

from .constituents import CATALOG
from .design import build_design_matrix, observation_weights
from .time import normalize_time
from .types import PhaseFitResult

EARTH_ROTATION_RAD_S = 7.2921159e-5


def coriolis_parameter(latitude_deg: np.ndarray | float) -> np.ndarray:
    return 2.0 * EARTH_ROTATION_RAD_S * np.sin(np.deg2rad(latitude_deg))


def current_aware_wavenumber(
    omega: float,
    coriolis: np.ndarray | float,
    phase_speed_m_s: float,
    along_current_m_s: np.ndarray | float,
) -> np.ndarray:
    """Solve the positive intrinsic-frequency root of the Doppler dispersion relation."""

    if phase_speed_m_s <= 0:
        raise ValueError("phase_speed_m_s must be positive")
    f, current = np.broadcast_arrays(
        np.asarray(coriolis, dtype=float), np.asarray(along_current_m_s, dtype=float)
    )
    ff = np.abs(f)
    admissible = np.isfinite(ff) & np.isfinite(current) & (omega > ff)
    a = current**2 - phase_speed_m_s**2
    b = -2.0 * omega * current
    c = omega**2 - ff**2
    discriminant = b**2 - 4.0 * a * c
    k0 = np.full(f.shape, np.nan)
    k0[admissible] = np.sqrt(c[admissible]) / phase_speed_m_s
    candidates = np.full((3, *f.shape), np.nan)
    linear = admissible & (np.abs(a) < 1.0e-14) & (np.abs(b) >= 1.0e-14)
    candidates[0, linear] = -c[linear] / b[linear]
    quadratic = admissible & ~linear & (np.abs(a) >= 1.0e-14) & (discriminant >= 0.0)
    root = np.zeros(f.shape)
    root[quadratic] = np.sqrt(discriminant[quadratic])
    candidates[1, quadratic] = (-b[quadratic] + root[quadratic]) / (2.0 * a[quadratic])
    candidates[2, quadratic] = (-b[quadratic] - root[quadratic]) / (2.0 * a[quadratic])
    sigma = omega - candidates * current[None, ...]
    valid = (
        np.isfinite(candidates)
        & (candidates > 0.0)
        & (sigma > ff[None, ...])
        & admissible[None, ...]
    )
    distance = np.where(valid, np.abs(candidates - k0[None, ...]), np.inf)
    choice = np.argmin(distance, axis=0)
    selected = np.take_along_axis(candidates, choice[None, ...], axis=0)[0]
    return np.where(np.all(~valid, axis=0), np.nan, selected)


@dataclass(frozen=True)
class PlaneWavePhase:
    kx: Mapping[str, float]
    ky: Mapping[str, float]
    x0: float
    y0: float

    def offsets(self, constituents, time, x, y, reference_time=None):
        if x is None or y is None:
            raise ValueError("PlaneWavePhase requires x and y")
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)
        return np.stack(
            [
                -float(self.kx.get(name, 0.0)) * (xx - self.x0)
                - float(self.ky.get(name, 0.0)) * (yy - self.y0)
                for name in constituents
            ]
        )


@dataclass(frozen=True)
class RegularGridCurrent:
    """A model-neutral two-dimensional background-current field."""

    time_seconds: np.ndarray
    y: np.ndarray
    x: np.ndarray
    u: np.ndarray
    v: np.ndarray

    def __post_init__(self):
        t = np.asarray(self.time_seconds, dtype=float)
        y = np.asarray(self.y, dtype=float)
        x = np.asarray(self.x, dtype=float)
        expected = (t.size, y.size, x.size)
        if np.asarray(self.u).shape != expected or np.asarray(self.v).shape != expected:
            raise ValueError(f"u and v must have shape {expected}")
        if np.any(np.diff(t) <= 0) or np.any(np.diff(y) <= 0) or np.any(np.diff(x) <= 0):
            raise ValueError("current coordinates must be strictly increasing")

    def sample(self, time_seconds, x, y):
        time_values, x_values, y_values = np.broadcast_arrays(
            np.asarray(time_seconds, dtype=float),
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )
        shape = time_values.shape
        points = np.column_stack(
            [
                time_values.reshape(-1),
                y_values.reshape(-1),
                x_values.reshape(-1),
            ]
        )
        kwargs = {"bounds_error": False, "fill_value": np.nan}
        grid = (self.time_seconds, self.y, self.x)
        return (
            RegularGridInterpolator(grid, self.u, **kwargs)(points).reshape(shape),
            RegularGridInterpolator(grid, self.v, **kwargs)(points).reshape(shape),
        )


@dataclass(frozen=True)
class StraightRayEikonalPhase:
    """Straight, parallel-ray phase integration through a background current."""

    direction_deg: Mapping[str, float]
    phase_speed_m_s: Mapping[str, float]
    reference_position: tuple[float, float]
    coriolis_rad_s: float
    current: RegularGridCurrent | None = None
    current_reference_time: np.datetime64 = np.datetime64("1970-01-01T00:00:00", "ns")
    integration_step_m: float = 2000.0

    def _one(self, name, time, x, y, reference_time):
        if name not in self.direction_deg or name not in self.phase_speed_m_s:
            return np.zeros(np.asarray(x).shape)
        omega = CATALOG[name].frequency_rad_s
        bearing = np.deg2rad(float(self.direction_deg[name]))
        hx, hy = np.sin(bearing), np.cos(bearing)
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)
        tt, ref = normalize_time(np.asarray(time), reference_time)
        current_shift = float((ref - self.current_reference_time) / np.timedelta64(1, "s"))
        tt = tt + current_shift
        xx, yy, tt = np.broadcast_arrays(xx, yy, tt)
        original_shape = xx.shape
        flat_x, flat_y, flat_t = xx.reshape(-1), yy.reshape(-1), tt.reshape(-1)
        x0, y0 = self.reference_position
        distance = (flat_x - x0) * hx + (flat_y - y0) * hy
        finite = np.isfinite(flat_x) & np.isfinite(flat_y) & np.isfinite(flat_t)
        output = np.full(flat_x.size, np.nan)
        speed = float(self.phase_speed_m_s[name])
        if self.current is None:
            k = current_aware_wavenumber(
                omega, self.coriolis_rad_s, speed, 0.0
            )
            output[finite] = -float(k) * distance[finite]
            return output.reshape(original_shape)
        if not np.any(finite):
            return output.reshape(original_shape)
        valid_index = np.flatnonzero(finite)
        valid_distance = distance[finite]
        max_distance = float(np.max(np.abs(valid_distance)))
        segments = max(1, int(np.ceil(max_distance / self.integration_step_m)))
        fraction = np.linspace(0.0, 1.0, segments + 1)[None, :]
        q = valid_distance[:, None] * fraction
        path_x = (flat_x[finite] - valid_distance * hx)[:, None] + q * hx
        path_y = (flat_y[finite] - valid_distance * hy)[:, None] + q * hy
        path_time = np.broadcast_to(flat_t[finite, None], path_x.shape)
        u, v = self.current.sample(path_time, path_x, path_y)
        along = u * hx + v * hy
        k = current_aware_wavenumber(
            omega, self.coriolis_rad_s, speed, along
        )
        supported = np.all(np.isfinite(k), axis=1)
        integral = -trapezoid(k, q, axis=1)
        output[valid_index[supported]] = integral[supported]
        return output.reshape(original_shape)

    def offsets(self, constituents, time, x, y, reference_time=None):
        if x is None or y is None:
            raise ValueError("StraightRayEikonalPhase requires x and y")
        return np.stack(
            [self._one(name, time, x, y, reference_time) for name in constituents]
        )


def fit_straight_ray_phase(
    *,
    time: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    constituent: str,
    reference_time: str | np.datetime64,
    reference_position: tuple[float, float],
    coriolis_rad_s: float,
    current: RegularGridCurrent | None = None,
    current_reference_time: np.datetime64 | None = None,
    direction_bounds: tuple[float, float] = (0.0, 360.0),
    phase_speed_bounds: tuple[float, float] = (0.25, 5.0),
    integration_step_m: float = 2000.0,
    background: str = "linear_space_time",
    weights: str | np.ndarray = "profile",
    group: np.ndarray | None = None,
    seed: int = 20191003,
) -> PhaseFitResult:
    """Fit bearing and effective phase speed by nonlinear variable projection."""

    name = constituent.upper()
    if current_reference_time is None:
        current_reference_time = np.datetime64("1970-01-01", "ns")
    if name not in CATALOG:
        raise ValueError(f"Unknown constituent {constituent!r}")
    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("values must have shape (observation,) or (observation, depth)")
    def broadcast_coordinate(raw, label, dtype=None):
        array = np.asarray(raw, dtype=dtype)
        if array.shape == (data.shape[0],):
            return np.broadcast_to(array[:, None], data.shape)
        if array.shape == data.shape:
            return array
        raise ValueError(f"{label} must have shape (observation,) or match values")

    t = broadcast_coordinate(time, "time")
    xx = broadcast_coordinate(x, "x", float)
    yy = broadcast_coordinate(y, "y", float)
    time_seconds, ref = normalize_time(t, reference_time)
    group_values = None if group is None else np.asarray(group).reshape(-1)
    if group_values is not None and group_values.size != data.shape[0]:
        raise ValueError("group must have one label per observation/profile")
    omega = CATALOG[name].frequency_rad_s
    standardized = data.copy()
    for iz in range(data.shape[1]):
        finite = np.isfinite(data[:, iz])
        scale = np.nanstd(data[finite, iz]) if np.any(finite) else np.nan
        if np.isfinite(scale) and scale > 0:
            standardized[:, iz] /= scale

    evaluations: list[tuple[float, float, float]] = []

    def objective(params):
        direction, speed = float(params[0]) % 360.0, float(params[1])
        model = StraightRayEikonalPhase(
            {name: direction},
            {name: speed},
            reference_position,
            coriolis_rad_s,
            current,
            current_reference_time,
            integration_step_m,
        )
        phase = model.offsets((name,), t, xx, yy, ref)[0]
        theta = omega * time_seconds + phase
        total, count = 0.0, 0
        for iz in range(standardized.shape[1]):
            response = standardized[:, iz]
            theta_column = theta[:, iz]
            time_column = time_seconds[:, iz]
            valid = (
                np.isfinite(response)
                & np.isfinite(theta_column)
                & np.isfinite(time_column)
            )
            if np.count_nonzero(valid) < 6:
                continue
            design = build_design_matrix(
                time_column[valid],
                np.asarray([omega]),
                phase_offset=phase[None, valid, iz],
                x=xx[valid, iz],
                y=yy[valid, iz],
                background=background,
            )
            group_valid = None if group_values is None else group_values[valid]
            sample_weight = observation_weights(
                weights, np.count_nonzero(valid), group_valid
            )
            root_weight = np.sqrt(sample_weight)
            coefficient, *_ = np.linalg.lstsq(
                design.values * root_weight[:, None],
                response[valid] * root_weight,
                rcond=None,
            )
            residual = response[valid] - design.values @ coefficient
            total += float(np.sum(sample_weight * residual**2))
            count += residual.size
        value = np.inf if count == 0 else total / count
        evaluations.append((direction, speed, float(value)))
        return value

    bounds = [direction_bounds, phase_speed_bounds]
    coarse = differential_evolution(
        objective, bounds, seed=seed, maxiter=20, popsize=8, polish=False, workers=1
    )
    refined = minimize(
        objective,
        coarse.x,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 300},
    )
    params = refined.x if refined.fun <= coarse.fun else coarse.x
    value = min(float(refined.fun), float(coarse.fun))
    success = bool(np.isfinite(value))
    evaluation_array = np.asarray(evaluations, dtype=float)
    return PhaseFitResult(
        constituent=name,
        direction_deg=float(params[0] % 360.0),
        phase_speed_m_s=float(params[1]),
        objective=value,
        success=success,
        n_observations=int(np.count_nonzero(np.any(np.isfinite(data), axis=1))),
        message=str(refined.message if refined.fun <= coarse.fun else coarse.message),
        metadata={
            "method": "variable_projection_straight_ray",
            "current_enabled": current is not None,
            "background": background,
            "weights": weights if isinstance(weights, str) else "custom",
            "objective_sample_direction_deg": evaluation_array[:, 0].tolist(),
            "objective_sample_phase_speed_m_s": evaluation_array[:, 1].tolist(),
            "objective_sample_value": evaluation_array[:, 2].tolist(),
        },
    )
