"""Depth-wise weighted harmonic regression."""

from __future__ import annotations

import warnings as python_warnings

import numpy as np

from .constituents import get_constituents
from .design import build_design_matrix, observation_weights
from .time import normalize_time
from .types import HarmonicResult

PHASE_CONVENTION = "C=A*exp(-i*phi); prediction=Re(C*exp(i*(omega*(t-t0)+psi)))"


def _as_profile_matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError("rho must have shape (observation,) or (observation, depth)")
    return array


def _broadcast_samples(values: np.ndarray | None, shape: tuple[int, int], name: str):
    if values is None:
        return None
    array = np.asarray(values)
    if array.shape == (shape[0],):
        return np.broadcast_to(array[:, None], shape)
    if array.shape == shape:
        return array
    raise ValueError(f"{name} must have shape (observation,) or match rho")


def _reference_matrix(
    rho_reference: np.ndarray | None,
    rho_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if rho_reference is None:
        return np.zeros(rho_shape), np.full(rho_shape[1], np.nan)
    ref = np.asarray(rho_reference, dtype=float)
    if ref.shape == (rho_shape[1],):
        matrix = np.broadcast_to(ref[None, :], rho_shape)
    elif ref.shape == rho_shape:
        matrix = ref
    else:
        raise ValueError("rho_reference must have shape (depth,) or match rho")
    return matrix, np.nanmean(matrix, axis=0)


def solve(
    *,
    time: np.ndarray,
    depth: np.ndarray,
    rho: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    constituents: tuple[str, ...] = ("M2", "K1"),
    rho_reference: np.ndarray | None = None,
    background: str = "linear_space_time",
    phase_offset: np.ndarray | None = None,
    phase_model=None,
    weights: str | np.ndarray = "profile",
    group: np.ndarray | None = None,
    reference_time: str | np.datetime64 | None = None,
    displacement: bool = True,
    min_density_gradient: float = 1.0e-5,
    uncertainty: str = "analytic",
) -> HarmonicResult:
    """Fit coherent harmonic coefficients independently at each depth.

    The sample axis is first. Profile-level or sample-level time and position arrays are
    accepted. The returned complex coefficient follows ``C=A exp(-i phi)``.
    """

    if uncertainty != "analytic":
        raise ValueError(
            "solve currently provides analytic uncertainty; use block_bootstrap separately"
        )
    matrix = _as_profile_matrix(rho)
    n_obs, n_depth = matrix.shape
    depth_values = np.asarray(depth, dtype=float).reshape(-1)
    if depth_values.size != n_depth or np.any(np.diff(depth_values) <= 0):
        raise ValueError("depth must be strictly increasing and match rho's depth axis")
    time_matrix = _broadcast_samples(np.asarray(time), matrix.shape, "time")
    x_matrix = _broadcast_samples(x, matrix.shape, "x")
    y_matrix = _broadcast_samples(y, matrix.shape, "y")
    group_values = None if group is None else np.asarray(group).reshape(-1)
    if group_values is not None and group_values.size != n_obs:
        raise ValueError("group must have one label per observation/profile")
    entries = get_constituents(constituents)
    names = tuple(item.name for item in entries)
    frequencies = np.asarray([item.frequency_rad_s for item in entries])
    all_time_seconds, ref_time = normalize_time(time_matrix, reference_time)
    if phase_model is not None:
        if phase_offset is not None:
            raise ValueError("provide phase_offset or phase_model, not both")
        phase_offset = phase_model.offsets(names, time_matrix, x_matrix, y_matrix, ref_time)
    if phase_offset is not None:
        phase_values = np.asarray(phase_offset, dtype=float)
        expected = (len(entries), n_obs, n_depth)
        if phase_values.shape == (len(entries), n_obs):
            phase_values = np.broadcast_to(phase_values[:, :, None], expected)
        if phase_values.shape != expected:
            raise ValueError(f"phase_offset must have shape {expected} or {(len(entries), n_obs)}")
    else:
        phase_values = np.zeros((len(entries), n_obs, n_depth))

    ref_matrix, background_density = _reference_matrix(rho_reference, matrix.shape)
    anomaly = matrix - ref_matrix if rho_reference is not None else matrix.copy()
    if rho_reference is not None:
        density_gradient = np.gradient(background_density, depth_values, edge_order=1)
    else:
        density_gradient = np.full(n_depth, np.nan)

    shape = (len(entries), n_depth)
    complex_density = np.full(shape, np.nan + 1j * np.nan)
    complex_displacement = np.full(shape, np.nan + 1j * np.nan)
    standard_error = np.full(shape, np.nan)
    harmonic_variance = np.full(shape, np.nan)
    variance_explained = np.full(n_depth, np.nan)
    valid_count = np.zeros(n_depth, dtype=int)
    design_rank = np.zeros(n_depth, dtype=int)
    design_condition = np.full(n_depth, np.nan)
    phase_coverage = np.full(shape, np.nan)
    background_coefficients: list[np.ndarray] = []
    background_terms: tuple[str, ...] = ()
    warning_codes: set[str] = set()

    for iz in range(n_depth):
        t = all_time_seconds[:, iz]
        xx = None if x_matrix is None else np.asarray(x_matrix[:, iz], dtype=float)
        yy = None if y_matrix is None else np.asarray(y_matrix[:, iz], dtype=float)
        valid = np.isfinite(anomaly[:, iz]) & np.isfinite(t)
        if xx is not None:
            valid &= np.isfinite(xx)
        if yy is not None:
            valid &= np.isfinite(yy)
        valid &= np.all(np.isfinite(phase_values[:, :, iz]), axis=0)
        if np.count_nonzero(valid) < 2 * len(entries) + 2:
            background_coefficients.append(np.asarray([], dtype=float))
            warning_codes.add("INSUFFICIENT_SUPPORT")
            continue
        t_valid = t[valid]
        design = build_design_matrix(
            t_valid,
            frequencies,
            phase_offset=phase_values[:, valid, iz],
            x=None if xx is None else xx[valid],
            y=None if yy is None else yy[valid],
            background=background,
        )
        response = anomaly[valid, iz]
        group_valid = None if group_values is None else group_values[valid]
        w = observation_weights(weights, response.size, group_valid)
        root_w = np.sqrt(w)
        aw = design.values * root_w[:, None]
        bw = response * root_w
        coefficient, _, rank, singular = np.linalg.lstsq(aw, bw, rcond=None)
        fitted = design.values @ coefficient
        residual = response - fitted
        dof = max(1, response.size - rank)
        sigma2 = float(np.sum(w * residual**2) / dof)
        covariance = sigma2 * np.linalg.pinv(aw.T @ aw)
        total = float(np.sum(w * (response - np.average(response, weights=w)) ** 2))
        variance_explained[iz] = np.nan if total <= 0 else 1.0 - np.sum(w * residual**2) / total
        valid_count[iz] = response.size
        design_rank[iz] = int(rank)
        design_condition[iz] = (
            np.inf if singular.size == 0 or singular[-1] == 0 else float(singular[0] / singular[-1])
        )
        background_terms = design.background_terms
        background_coefficients.append(coefficient[: design.harmonic_start])
        if rank < design.values.shape[1]:
            warning_codes.add("RANK_DEFICIENT")
        if design_condition[iz] > 1.0e8:
            warning_codes.add("ILL_CONDITIONED")
        for ic in range(len(entries)):
            j = design.harmonic_start + 2 * ic
            value = coefficient[j] + 1j * coefficient[j + 1]
            complex_density[ic, iz] = value
            pair_variance = covariance[j, j] + covariance[j + 1, j + 1]
            standard_error[ic, iz] = float(np.sqrt(max(0.0, pair_variance)))
            harmonic_variance[ic, iz] = 0.5 * abs(value) ** 2
            phase_coverage[ic, iz] = 1.0 - abs(np.mean(np.exp(1j * design.theta[ic])))

    if background_coefficients:
        width = max((item.size for item in background_coefficients), default=0)
        background_array = np.full((width, n_depth), np.nan)
        for iz, item in enumerate(background_coefficients):
            background_array[: item.size, iz] = item
    else:
        background_array = np.empty((0, n_depth))
    if displacement and rho_reference is not None:
        strong = np.isfinite(density_gradient) & (density_gradient >= min_density_gradient)
        complex_displacement[:, strong] = complex_density[:, strong] / density_gradient[strong]
        if np.any(~strong):
            warning_codes.add("WEAK_STRATIFICATION")
    amplitude_density = np.abs(complex_density)
    amplitude_displacement = np.abs(complex_displacement)
    phase_lag = -np.angle(complex_density)

    finite_time = np.asarray(all_time_seconds)[np.isfinite(all_time_seconds)]
    record_seconds = float(np.ptp(finite_time)) if finite_time.size else 0.0
    resolved = np.ones(len(entries), dtype=bool)
    if len(entries) > 1 and record_seconds > 0:
        for i, frequency in enumerate(frequencies):
            separation = np.min(np.abs(np.delete(frequencies, i) - frequency))
            resolved[i] = separation >= 2.0 * np.pi / record_seconds
    if not np.all(resolved):
        warning_codes.add("UNRESOLVED_CONSTITUENTS")
    for code in sorted(warning_codes):
        python_warnings.warn(code, RuntimeWarning, stacklevel=2)
    return HarmonicResult(
        constituents=names,
        frequency_rad_s=frequencies,
        depth=depth_values,
        reference_time=ref_time,
        phase_convention=PHASE_CONVENTION,
        complex_density=complex_density,
        complex_displacement=complex_displacement,
        amplitude_density=amplitude_density,
        amplitude_displacement=amplitude_displacement,
        phase_lag=phase_lag,
        standard_error_density=standard_error,
        harmonic_variance=harmonic_variance,
        background_density=background_density,
        density_gradient=density_gradient,
        background_coefficients=background_array,
        background_terms=background_terms,
        valid_count=valid_count,
        variance_explained=variance_explained,
        phase_coverage=phase_coverage,
        design_rank=design_rank,
        design_condition=design_condition,
        resolved_constituent=resolved,
        warnings=tuple(sorted(warning_codes)),
        metadata={"depth_positive": "down", "displacement_positive": "up"},
    )
