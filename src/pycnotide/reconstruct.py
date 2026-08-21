"""Harmonic displacement and density reconstruction."""

from __future__ import annotations

import numpy as np

from .time import normalize_time
from .types import HarmonicResult, Prediction


def _interp_complex(depth_in, values, depth_out):
    output = np.full((values.shape[0], depth_out.size), np.nan + 1j * np.nan)
    for row, source in enumerate(values):
        valid = np.isfinite(source)
        if np.count_nonzero(valid) >= 2:
            real = np.interp(
                depth_out, depth_in[valid], source[valid].real, left=np.nan, right=np.nan
            )
            imag = np.interp(
                depth_out, depth_in[valid], source[valid].imag, left=np.nan, right=np.nan
            )
            output[row] = real + 1j * imag
    return output


def reconstruct(
    harmonics: HarmonicResult,
    *,
    time: np.ndarray,
    depth: np.ndarray | None = None,
    rho_reference: np.ndarray | None = None,
    phase_offset: np.ndarray | None = None,
    method: str = "isopycnal_remap",
    return_components: bool = True,
) -> Prediction:
    """Reconstruct displacement, isopycnal velocity, and density.

    ``time`` may be an ordinary one-dimensional series or an exact-track matrix
    shaped ``(observation, depth)``. Returned scientific arrays always use
    ``(depth, observation)`` after any leading constituent dimension.
    """

    del return_components
    output_depth = harmonics.depth if depth is None else np.asarray(depth, dtype=float).reshape(-1)
    if np.any(np.diff(output_depth) <= 0):
        raise ValueError("depth must be strictly increasing")
    time_values = np.asarray(time)
    if time_values.ndim not in {1, 2}:
        raise ValueError("time must have shape (observation,) or (observation, depth)")
    if time_values.ndim == 2 and time_values.shape[1] != output_depth.size:
        raise ValueError("two-dimensional time must have shape (observation, depth)")
    time_seconds, _ = normalize_time(time_values, harmonics.reference_time)
    observation_count = time_values.shape[0]
    coefficients = _interp_complex(
        harmonics.depth, harmonics.complex_displacement, output_depth
    )
    if not np.any(np.isfinite(coefficients)):
        raise ValueError("harmonics contain no finite displacement coefficients")
    constituent_count = len(harmonics.constituents)
    if time_values.ndim == 1:
        temporal_angle = harmonics.frequency_rad_s[:, None, None] * time_seconds[None, None, :]
        if phase_offset is None:
            phase = np.zeros((constituent_count, output_depth.size, observation_count))
        else:
            raw_phase = np.asarray(phase_offset, dtype=float)
            if raw_phase.shape == (constituent_count, observation_count):
                phase = np.broadcast_to(
                    raw_phase[:, None, :],
                    (constituent_count, output_depth.size, observation_count),
                )
            elif raw_phase.shape == (
                constituent_count,
                output_depth.size,
                observation_count,
            ):
                phase = raw_phase
            else:
                raise ValueError("phase_offset has an incompatible time-series shape")
    else:
        temporal_angle = (
            harmonics.frequency_rad_s[:, None, None]
            * time_seconds.T[None, :, :]
        )
        if phase_offset is None:
            phase = np.zeros((constituent_count, output_depth.size, observation_count))
        else:
            raw_phase = np.asarray(phase_offset, dtype=float)
            if raw_phase.shape == (constituent_count, observation_count, output_depth.size):
                phase = raw_phase.transpose(0, 2, 1)
            elif raw_phase.shape == (
                constituent_count,
                output_depth.size,
                observation_count,
            ):
                phase = raw_phase
            else:
                raise ValueError("phase_offset has an incompatible exact-track shape")
    carrier = np.exp(1j * (temporal_angle + phase))
    xi_by = np.real(coefficients[:, :, None] * carrier)
    w_by = np.real(
        1j
        * harmonics.frequency_rad_s[:, None, None]
        * coefficients[:, :, None]
        * carrier
    )
    harmonic_support = np.any(np.isfinite(xi_by), axis=0)
    xi = np.nansum(xi_by, axis=0)
    w_iso = np.nansum(w_by, axis=0)
    xi[~harmonic_support] = np.nan
    w_iso[~harmonic_support] = np.nan
    if rho_reference is None:
        reference_profile = np.interp(
            output_depth, harmonics.depth, harmonics.background_density, left=np.nan, right=np.nan
        )
        reference = np.broadcast_to(
            reference_profile[:, None], (output_depth.size, observation_count)
        )
    else:
        raw_reference = np.asarray(rho_reference, dtype=float)
        if raw_reference.shape == (output_depth.size,):
            reference = np.broadcast_to(
                raw_reference[:, None], (output_depth.size, observation_count)
            )
        elif raw_reference.shape == (observation_count, output_depth.size):
            reference = raw_reference.T
        elif raw_reference.shape == (output_depth.size, observation_count):
            reference = raw_reference
        else:
            raise ValueError("rho_reference must match depth or the exact-track matrix")
    gradient = np.gradient(reference, output_depth, axis=0, edge_order=1)
    rho_anomaly_by = xi_by * gradient[None, :, :]
    rho_anomaly = np.nansum(rho_anomaly_by, axis=0)
    rho_anomaly[~harmonic_support] = np.nan
    if method == "linear":
        rho = reference + rho_anomaly
        mask = np.isfinite(rho)
    elif method == "isopycnal_remap":
        rho = np.full_like(xi, np.nan)
        mask = np.zeros_like(xi, dtype=bool)
        for it in range(observation_count):
            displaced_depth = output_depth - xi[:, it]
            valid = np.isfinite(displaced_depth) & np.isfinite(reference[:, it])
            if np.count_nonzero(valid) < 2 or np.any(np.diff(displaced_depth[valid]) <= 0):
                continue
            rho[:, it] = np.interp(
                output_depth,
                displaced_depth[valid],
                reference[valid, it],
                left=np.nan,
                right=np.nan,
            )
            mask[:, it] = np.isfinite(rho[:, it])
    else:
        raise ValueError("method must be 'linear' or 'isopycnal_remap'")
    return Prediction(
        time=time_values,
        depth=output_depth,
        xi=xi,
        xi_by_constituent=xi_by,
        w_iso=w_iso,
        w_iso_by_constituent=w_by,
        rho=rho,
        rho_anomaly=rho_anomaly,
        rho_anomaly_by_constituent=rho_anomaly_by,
        mask=mask,
        metadata={
            "method": method,
            "phase_convention": harmonics.phase_convention,
            "array_layout": "depth,observation",
            "exact_track": time_values.ndim == 2,
        },
    )
