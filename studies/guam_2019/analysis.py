"""Harmonic fitting, phase sensitivity, reconstruction, rendering, and QA."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

import pycnotide
from pycnotide.time import as_datetime64_ns, normalize_time
from pycnotide.types import HarmonicResult

from .acquisition import file_checksum, file_sha256, write_json


def _imports():
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - guarded by preflight
        raise RuntimeError("Install pycnotide[guam] for campaign analysis") from exc
    return xr, pd, plt


def _paths(config: dict, repository: Path) -> tuple[Path, Path]:
    processed = (repository / config["paths"]["processed"]).resolve()
    outputs = (repository / config["paths"]["outputs"]).resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    return processed, outputs


def _phase_fit_depth_mask(
    depth: np.ndarray, lower_m: float, upper_m: float, stride_m: float
) -> np.ndarray:
    """Select deterministic native levels without altering their values."""

    values = np.asarray(depth, dtype=float)
    if values.ndim != 1 or stride_m <= 0:
        raise ValueError("depth must be one-dimensional and stride_m must be positive")
    candidates = np.flatnonzero(
        np.isfinite(values) & (values >= lower_m) & (values <= upper_m)
    )
    mask = np.zeros(values.size, dtype=bool)
    if candidates.size == 0:
        return mask
    last = -np.inf
    tolerance = max(1.0, abs(stride_m)) * 1.0e-10
    for index in candidates:
        if values[index] - last >= stride_m - tolerance:
            mask[index] = True
            last = values[index]
    return mask


def _result_dataset(result: HarmonicResult):
    xr, _, _ = _imports()
    coords = {"constituent": list(result.constituents), "depth": result.depth}
    dataset = xr.Dataset(
        {
            "complex_density_real": (
                ("constituent", "depth"),
                result.complex_density.real,
            ),
            "complex_density_imag": (
                ("constituent", "depth"),
                result.complex_density.imag,
            ),
            "complex_displacement_real": (
                ("constituent", "depth"),
                result.complex_displacement.real,
            ),
            "complex_displacement_imag": (
                ("constituent", "depth"),
                result.complex_displacement.imag,
            ),
            "amplitude_density": (("constituent", "depth"), result.amplitude_density),
            "amplitude_displacement": (
                ("constituent", "depth"),
                result.amplitude_displacement,
            ),
            "phase_lag": (("constituent", "depth"), result.phase_lag),
            "standard_error_density": (
                ("constituent", "depth"),
                result.standard_error_density,
            ),
            "harmonic_variance": (
                ("constituent", "depth"),
                result.harmonic_variance,
            ),
            "phase_coverage": (("constituent", "depth"), result.phase_coverage),
            "resolved_constituent": (
                ("constituent",),
                result.resolved_constituent,
            ),
            "background_density": (("depth",), result.background_density),
            "density_gradient": (("depth",), result.density_gradient),
            "valid_count": (("depth",), result.valid_count),
            "variance_explained": (("depth",), result.variance_explained),
            "design_rank": (("depth",), result.design_rank),
            "design_condition": (("depth",), result.design_condition),
            "frequency_rad_s": (("constituent",), result.frequency_rad_s),
        },
        coords=coords,
        attrs={
            "schema": "pycnotide_harmonics_v1",
            "reference_time": str(result.reference_time),
            "phase_convention": result.phase_convention,
            "warnings": ",".join(result.warnings),
            "depth_positive": "down",
            "displacement_positive": "up",
        },
    )
    return dataset


def _result_from_dataset(dataset) -> HarmonicResult:
    complex_density = (
        dataset.complex_density_real.values + 1j * dataset.complex_density_imag.values
    )
    complex_displacement = (
        dataset.complex_displacement_real.values
        + 1j * dataset.complex_displacement_imag.values
    )
    return HarmonicResult(
        constituents=tuple(str(value) for value in dataset.constituent.values),
        frequency_rad_s=dataset.frequency_rad_s.values,
        depth=dataset.depth.values,
        reference_time=np.datetime64(dataset.attrs["reference_time"], "ns"),
        phase_convention=dataset.attrs["phase_convention"],
        complex_density=complex_density,
        complex_displacement=complex_displacement,
        amplitude_density=dataset.amplitude_density.values,
        amplitude_displacement=dataset.amplitude_displacement.values,
        phase_lag=dataset.phase_lag.values,
        standard_error_density=dataset.standard_error_density.values,
        harmonic_variance=dataset.harmonic_variance.values,
        background_density=dataset.background_density.values,
        density_gradient=dataset.density_gradient.values,
        background_coefficients=np.empty((0, dataset.sizes["depth"])),
        background_terms=(),
        valid_count=dataset.valid_count.values,
        variance_explained=dataset.variance_explained.values,
        phase_coverage=dataset.phase_coverage.values,
        design_rank=dataset.design_rank.values,
        design_condition=dataset.design_condition.values,
        resolved_constituent=dataset.resolved_constituent.values,
        warnings=tuple(filter(None, dataset.attrs.get("warnings", "").split(","))),
        metadata={"loaded_from": "campaign NetCDF"},
    )


def _solve_dataset(ds, config: dict, indices: np.ndarray | slice = slice(None)):
    analysis = config["analysis"]
    return pycnotide.solve(
        time=ds.sample_time.values[indices],
        depth=ds.depth.values,
        rho=ds.density.values[indices],
        rho_reference=ds.density_reference.values[indices],
        x=ds.x.values[indices],
        y=ds.y.values[indices],
        constituents=tuple(analysis["constituents"]),
        background=analysis["background"],
        weights="group",
        group=ds.glider.values[indices],
        reference_time=analysis["reference_time"],
        min_density_gradient=analysis["min_density_gradient_kg_m4"],
    )


def fit_harmonics(config: dict, repository: Path) -> Path:
    xr, pd, _ = _imports()
    processed, outputs = _paths(config, repository)
    source = processed / "guam_seaglider_level3_processed.nc"
    with xr.open_dataset(source) as opened:
        ds = opened.load()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = _solve_dataset(ds, config)
    dataset = _result_dataset(result)

    profile_seconds = (
        ds.profile_time.values.astype("datetime64[ns]")
        - as_datetime64_ns(config["analysis"]["reference_time"])
    ) / np.timedelta64(1, "s")
    resamples = int(config["analysis"]["bootstrap_resamples"])
    indices = pycnotide.block_bootstrap_indices(
        profile_seconds,
        block_seconds=float(config["analysis"]["bootstrap_block_hours"]) * 3600.0,
        n_resamples=resamples,
        seed=int(config["analysis"]["bootstrap_seed"]),
    )
    boot = np.full(
        (resamples, len(result.constituents), result.depth.size),
        np.nan + 1j * np.nan,
        dtype=np.complex64,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, sample in enumerate(indices):
            boot[i] = _solve_dataset(ds, config, sample).complex_displacement
    amplitude = np.abs(boot)
    dataset["bootstrap_amplitude_p025"] = (
        ("constituent", "depth"),
        np.nanpercentile(amplitude, 2.5, axis=0),
    )
    dataset["bootstrap_amplitude_p975"] = (
        ("constituent", "depth"),
        np.nanpercentile(amplitude, 97.5, axis=0),
    )
    dataset.attrs["bootstrap_resamples"] = resamples
    dataset.attrs["bootstrap_block_hours"] = config["analysis"]["bootstrap_block_hours"]
    output = outputs / "guam2019_harmonics.nc"
    dataset.to_netcdf(
        output,
        engine="netcdf4",
        encoding={name: {"zlib": True} for name in dataset.data_vars},
    )

    rows: list[dict] = []
    for ic, name in enumerate(result.constituents):
        for iz, depth in enumerate(result.depth):
            rows.append(
                {
                    "constituent": name,
                    "depth_m": depth,
                    "density_real_kg_m3": result.complex_density[ic, iz].real,
                    "density_imag_kg_m3": result.complex_density[ic, iz].imag,
                    "displacement_real_m": result.complex_displacement[ic, iz].real,
                    "displacement_imag_m": result.complex_displacement[ic, iz].imag,
                    "amplitude_displacement_m": result.amplitude_displacement[ic, iz],
                    "phase_lag_rad": result.phase_lag[ic, iz],
                    "bootstrap_amplitude_p025_m": dataset.bootstrap_amplitude_p025.values[
                        ic, iz
                    ],
                    "bootstrap_amplitude_p975_m": dataset.bootstrap_amplitude_p975.values[
                        ic, iz
                    ],
                    "valid_count": result.valid_count[iz],
                    "variance_explained": result.variance_explained[iz],
                }
            )
    pd.DataFrame(rows).to_csv(outputs / "harmonic_profile_summary.csv", index=False)
    _fit_windows(ds, config, outputs)
    return output


def _fit_windows(ds, config: dict, outputs: Path) -> None:
    _, pd, _ = _imports()
    time = ds.profile_time.values.astype("datetime64[ns]")
    finite = ~np.isnat(time)
    start, end = np.min(time[finite]), np.max(time[finite])
    width = np.timedelta64(int(config["analysis"]["window_days"]), "D")
    step = np.timedelta64(int(config["analysis"]["window_step_days"]), "D")
    rows: list[dict] = []
    cursor = start
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        while cursor + width <= end + np.timedelta64(1, "D"):
            selected = np.flatnonzero((time >= cursor) & (time < cursor + width))
            if selected.size >= 20:
                result = _solve_dataset(ds, config, selected)
                for ic, name in enumerate(result.constituents):
                    rows.append(
                        {
                            "window_start": str(cursor),
                            "window_end": str(cursor + width),
                            "constituent": name,
                            "profile_count": selected.size,
                            "median_amplitude_displacement_m": np.nanmedian(
                                result.amplitude_displacement[ic]
                            ),
                            "median_variance_explained": np.nanmedian(
                                result.variance_explained
                            ),
                            "resolved": bool(result.resolved_constituent[ic]),
                        }
                    )
            cursor += step
    pd.DataFrame(rows).to_csv(outputs / "window_harmonic_summary.csv", index=False)


def _load_current(processed: Path, reference_time: np.datetime64):
    xr, _, _ = _imports()
    source = processed / "hycom_background_current_epsg32655.nc"
    with xr.open_dataset(source) as opened:
        ds = opened.load()
    seconds = (ds.time.values.astype("datetime64[ns]") - reference_time) / np.timedelta64(
        1, "s"
    )
    return pycnotide.RegularGridCurrent(
        np.asarray(seconds, dtype=float),
        ds.y.values,
        ds.x.values,
        ds.u_background.values,
        ds.v_background.values,
    )


def _conditional_block_skill(
    values,
    time,
    phase,
    omega,
    profile_time,
    block_hours=48.0,
):
    time_seconds, _ = normalize_time(time, np.datetime64("2019-10-03"))
    blocks = np.floor(
        (profile_time - np.nanmin(profile_time)) / (block_hours * 3600.0)
    ).astype(int)
    fold = blocks % 5
    squared, count = 0.0, 0
    for held in range(5):
        train_profile = fold != held
        test_profile = fold == held
        for iz in range(values.shape[1]):
            response = values[:, iz]
            theta = omega * time_seconds[:, iz] + phase[:, iz]
            train = train_profile & np.isfinite(response) & np.isfinite(theta)
            test = test_profile & np.isfinite(response) & np.isfinite(theta)
            if np.count_nonzero(train) < 6 or not np.any(test):
                continue
            design_train = np.column_stack(
                [np.ones(np.count_nonzero(train)), np.cos(theta[train]), -np.sin(theta[train])]
            )
            coefficient, *_ = np.linalg.lstsq(design_train, response[train], rcond=None)
            design_test = np.column_stack(
                [np.ones(np.count_nonzero(test)), np.cos(theta[test]), -np.sin(theta[test])]
            )
            residual = response[test] - design_test @ coefficient
            squared += float(np.sum(residual**2))
            count += residual.size
    return np.nan if count == 0 else float(np.sqrt(squared / count)), count


def _write_current_comparison(glider, current, reference_time, config, outputs) -> None:
    _, pd, _ = _imports()
    try:
        from pyproj import Proj
    except ImportError as exc:  # pragma: no cover - guarded by preflight
        raise RuntimeError("Install pycnotide[guam] for projected-current QA") from exc
    profile_seconds = (
        glider.profile_time.values.astype("datetime64[ns]") - reference_time
    ) / np.timedelta64(1, "s")
    x = np.nanmedian(glider.x.values, axis=1)
    y = np.nanmedian(glider.y.values, axis=1)
    hycom_u, hycom_v = current.sample(profile_seconds, x, y)
    projection = Proj(f"EPSG:{config['analysis']['epsg']}")
    convergence = np.deg2rad(
        projection.get_factors(
            glider.lon_dive.values, glider.lat_dive.values
        ).meridian_convergence
    )
    cosine, sine = np.cos(convergence), np.sin(convergence)
    glider_u = glider.u_dive.values * cosine - glider.v_dive.values * sine
    glider_v = glider.u_dive.values * sine + glider.v_dive.values * cosine
    frame = pd.DataFrame(
        {
            "profile": glider.profile.values,
            "time": glider.profile_time.values,
            "glider": glider.glider.values,
            "x_m": x,
            "y_m": y,
            "glider_u_projected_m_s": glider_u,
            "glider_v_projected_m_s": glider_v,
            "hycom_u_background_m_s": hycom_u,
            "hycom_v_background_m_s": hycom_v,
        }
    )
    frame.to_csv(outputs / "current_sampling_comparison.csv", index=False)
    rows = []
    for component in ("u", "v"):
        observed = frame[f"glider_{component}_projected_m_s"].to_numpy()
        modeled = frame[f"hycom_{component}_background_m_s"].to_numpy()
        valid = np.isfinite(observed) & np.isfinite(modeled)
        correlation = (
            np.corrcoef(observed[valid], modeled[valid])[0, 1]
            if np.count_nonzero(valid) >= 3
            else np.nan
        )
        difference = modeled[valid] - observed[valid]
        bias = float(np.mean(difference)) if difference.size else np.nan
        rmse = float(np.sqrt(np.mean(difference**2))) if difference.size else np.nan
        rows.append(
            {
                "component": component,
                "common_count": np.count_nonzero(valid),
                "hycom_minus_glider_bias_m_s": bias,
                "rmse_m_s": rmse,
                "correlation": correlation,
                "qualification": (
                    "approximate consistency only: unlike averaging volumes and times"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(outputs / "current_statistics.csv", index=False)


def fit_phase_sensitivity(config: dict, repository: Path) -> Path:
    xr, pd, _ = _imports()
    processed, outputs = _paths(config, repository)
    with xr.open_dataset(processed / "guam_seaglider_level3_processed.nc") as opened:
        glider = opened.load()
    reference_time = as_datetime64_ns(config["analysis"]["reference_time"])
    current = _load_current(processed, reference_time)
    depth = glider.depth.values
    fit_depth = _phase_fit_depth_mask(
        depth,
        config["analysis"]["phase_depth_min_m"],
        config["analysis"]["phase_depth_max_m"],
        config["analysis"].get("phase_fit_depth_stride_m", 1.0),
    )
    if np.count_nonzero(fit_depth) < 3:
        raise ValueError("Fewer than three native depth levels support phase fitting")
    values = glider.displacement.values[:, fit_depth]
    time = glider.sample_time.values[:, fit_depth]
    x = glider.x.values[:, fit_depth]
    y = glider.y.values[:, fit_depth]
    latitude = glider.latitude.values[:, fit_depth]
    reference_position = (float(np.nanmedian(x)), float(np.nanmedian(y)))
    coriolis = float(pycnotide.coriolis_parameter(np.nanmedian(latitude)))
    profile_seconds = (
        glider.profile_time.values.astype("datetime64[ns]") - reference_time
    ) / np.timedelta64(1, "s")
    _write_current_comparison(
        glider, current, reference_time, config, outputs
    )
    rows: list[dict] = []
    objective_rows: list[dict] = []
    scenarios: list[str] = []
    constituents: list[str] = []
    for scenario, field in (("zero_current", None), ("hycom_current", current)):
        for name in config["analysis"]["phase_constituents"]:
            fit = pycnotide.fit_straight_ray_phase(
                time=time,
                x=x,
                y=y,
                values=values,
                constituent=name,
                reference_time=reference_time,
                reference_position=reference_position,
                coriolis_rad_s=coriolis,
                current=field,
                current_reference_time=reference_time,
                direction_bounds=tuple(config["analysis"]["direction_bounds_deg"]),
                phase_speed_bounds=tuple(config["analysis"]["phase_speed_bounds_m_s"]),
                integration_step_m=config["analysis"]["phase_integration_step_m"],
                background=config["analysis"]["background"],
                weights="group",
                group=glider.glider.values,
                seed=config["analysis"]["bootstrap_seed"],
            )
            model = pycnotide.StraightRayEikonalPhase(
                {name: fit.direction_deg},
                {name: fit.phase_speed_m_s},
                reference_position,
                coriolis,
                field,
                reference_time,
                config["analysis"]["phase_integration_step_m"],
            )
            phase = model.offsets((name,), time, x, y, reference_time)[0]
            rmse, support = _conditional_block_skill(
                values,
                time,
                phase,
                pycnotide.CATALOG[name].frequency_rad_s,
                np.asarray(profile_seconds, dtype=float),
                config["analysis"]["bootstrap_block_hours"],
            )
            row = asdict(fit)
            metadata = row.pop("metadata")
            objective_rows.extend(
                {
                    "scenario": scenario,
                    "constituent": name,
                    "direction_deg": direction,
                    "phase_speed_m_s": speed,
                    "objective": objective,
                }
                for direction, speed, objective in zip(
                    metadata["objective_sample_direction_deg"],
                    metadata["objective_sample_phase_speed_m_s"],
                    metadata["objective_sample_value"],
                    strict=True,
                )
            )
            row.update(
                {
                    "scenario": scenario,
                    "conditional_block_cv_rmse_m": rmse,
                    "conditional_block_cv_count": support,
                }
            )
            rows.append(row)
            scenarios.append(scenario)
            constituents.append(name)
    frame = pd.DataFrame(rows)
    frame.to_csv(outputs / "phase_fit_summary.csv", index=False)
    pd.DataFrame(objective_rows).to_csv(
        outputs / "phase_objective_samples.csv", index=False
    )
    scenario_names = ("zero_current", "hycom_current")
    all_constituents = tuple(config["analysis"]["constituents"])
    scenario_phase = np.zeros(
        (
            len(scenario_names),
            len(all_constituents),
            glider.sizes["profile"],
            glider.sizes["depth"],
        ),
        dtype=np.float32,
    )
    fields = {"zero_current": None, "hycom_current": current}
    for row in frame.itertuples(index=False):
        scenario_index = scenario_names.index(row.scenario)
        constituent_index = all_constituents.index(row.constituent)
        model = pycnotide.StraightRayEikonalPhase(
            {row.constituent: row.direction_deg},
            {row.constituent: row.phase_speed_m_s},
            reference_position,
            coriolis,
            fields[row.scenario],
            reference_time,
            config["analysis"]["phase_integration_step_m"],
        )
        scenario_phase[scenario_index, constituent_index] = model.offsets(
            (row.constituent,),
            glider.sample_time.values,
            glider.x.values,
            glider.y.values,
            reference_time,
        )[0]
    phase_indices = [
        all_constituents.index(name)
        for name in config["analysis"]["phase_constituents"]
    ]
    shared_support = (
        np.isfinite(glider.density.values)
        & np.isfinite(glider.density_reference.values)
        & ~np.isnat(glider.sample_time.values)
        & np.isfinite(glider.x.values)
        & np.isfinite(glider.y.values)
        & np.all(np.isfinite(scenario_phase[:, phase_indices]), axis=(0, 1))
    )
    scenario_phase = np.where(
        shared_support[None, None, :, :], scenario_phase, np.nan
    )
    scenario_results: list[HarmonicResult] = []
    density = np.where(shared_support, glider.density.values, np.nan)
    density_reference = np.where(
        shared_support, glider.density_reference.values, np.nan
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for scenario_index in range(len(scenario_names)):
            scenario_results.append(
                pycnotide.solve(
                    time=glider.sample_time.values,
                    depth=glider.depth.values,
                    rho=density,
                    rho_reference=density_reference,
                    x=glider.x.values,
                    y=glider.y.values,
                    constituents=all_constituents,
                    background=config["analysis"]["background"],
                    phase_offset=scenario_phase[scenario_index],
                    weights="group",
                    group=glider.glider.values,
                    reference_time=reference_time,
                    min_density_gradient=config["analysis"][
                        "min_density_gradient_kg_m4"
                    ],
                )
            )
    case_phase = np.stack(
        [
            scenario_phase[
                scenario_names.index(scenario),
                all_constituents.index(constituent),
            ][:, fit_depth]
            for scenario, constituent in zip(scenarios, constituents, strict=True)
        ]
    )
    output = outputs / "guam2019_current_phase_sensitivity.nc"
    xr.Dataset(
        {
            "phase_offset": (
                ("case", "profile", "phase_depth"),
                case_phase,
            ),
            "scenario_phase_offset": (
                ("scenario_name", "harmonic_constituent", "profile", "depth"),
                scenario_phase,
            ),
            "scenario_complex_density_real": (
                ("scenario_name", "harmonic_constituent", "depth"),
                np.stack([result.complex_density.real for result in scenario_results]),
            ),
            "scenario_complex_density_imag": (
                ("scenario_name", "harmonic_constituent", "depth"),
                np.stack([result.complex_density.imag for result in scenario_results]),
            ),
            "scenario_complex_displacement_real": (
                ("scenario_name", "harmonic_constituent", "depth"),
                np.stack(
                    [result.complex_displacement.real for result in scenario_results]
                ),
            ),
            "scenario_complex_displacement_imag": (
                ("scenario_name", "harmonic_constituent", "depth"),
                np.stack(
                    [result.complex_displacement.imag for result in scenario_results]
                ),
            ),
            "shared_observational_support": (
                ("profile", "depth"),
                shared_support,
            ),
            "direction_deg": (("case",), frame.direction_deg.values),
            "phase_speed_m_s": (("case",), frame.phase_speed_m_s.values),
            "objective": (("case",), frame.objective.values),
            "conditional_block_cv_rmse_m": (
                ("case",),
                frame.conditional_block_cv_rmse_m.values,
            ),
        },
        coords={
            "case": np.arange(len(rows)),
            "scenario": (("case",), scenarios),
            "constituent": (("case",), constituents),
            "scenario_name": list(scenario_names),
            "harmonic_constituent": list(all_constituents),
            "profile": glider.profile.values,
            "depth": glider.depth.values,
            "phase_depth": depth[fit_depth],
        },
        attrs={
            "schema": "pycnotide_current_phase_sensitivity_v1",
            "reference_time": str(reference_time),
            "reference_x": reference_position[0],
            "reference_y": reference_position[1],
            "coriolis_rad_s": coriolis,
            "interpretation": "straight-ray eikonal sensitivity; not bent-ray tracing",
            "support_rule": (
                "zero-current and HYCOM-current harmonic products use identical "
                "finite observations"
            ),
            "phase_fit_depth_stride_m": config["analysis"].get(
                "phase_fit_depth_stride_m", 1.0
            ),
            "phase_fit_sampling": (
                "native depth levels selected at the configured spacing; exact sample "
                "times and underwater positions retained; final phase evaluated at all "
                "supported native levels"
            ),
        },
    ).to_netcdf(
        output,
        engine="netcdf4",
        encoding={
            "phase_offset": {"zlib": True},
            "scenario_phase_offset": {"zlib": True},
        },
    )
    return output


def reconstruct_products(config: dict, repository: Path) -> Path:
    xr, _, _ = _imports()
    processed, outputs = _paths(config, repository)
    with xr.open_dataset(outputs / "guam2019_harmonics.nc") as opened:
        result = _result_from_dataset(opened.load())
    with xr.open_dataset(processed / "guam_seaglider_level3_processed.nc") as opened:
        glider = opened.load()
    with xr.open_dataset(outputs / "guam2019_current_phase_sensitivity.nc") as opened:
        phase_product = opened.load()
    time_min = np.min(glider.profile_time.values[~np.isnat(glider.profile_time.values)])
    time_max = np.max(glider.profile_time.values[~np.isnat(glider.profile_time.values)])
    hourly = np.arange(time_min, time_max + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    rho_reference = np.nanmedian(glider.density_reference.values, axis=0)
    scenario_names = tuple(
        str(value) for value in phase_product.scenario_name.values
    )
    reference_linear = []
    reference_remap = []
    track_linear = []
    track_remap = []
    for scenario_index in range(len(scenario_names)):
        complex_density = (
            phase_product.scenario_complex_density_real.values[scenario_index]
            + 1j * phase_product.scenario_complex_density_imag.values[scenario_index]
        )
        complex_displacement = (
            phase_product.scenario_complex_displacement_real.values[scenario_index]
            + 1j
            * phase_product.scenario_complex_displacement_imag.values[scenario_index]
        )
        scenario_result = replace(
            result,
            complex_density=complex_density,
            complex_displacement=complex_displacement,
            amplitude_density=np.abs(complex_density),
            amplitude_displacement=np.abs(complex_displacement),
            phase_lag=-np.angle(complex_density),
            harmonic_variance=0.5 * np.abs(complex_density) ** 2,
            metadata={
                **result.metadata,
                "phase_scenario": scenario_names[scenario_index],
            },
        )
        reference_linear.append(
            pycnotide.reconstruct(
                scenario_result,
                time=hourly,
                depth=glider.depth.values,
                rho_reference=rho_reference,
                method="linear",
            )
        )
        reference_remap.append(
            pycnotide.reconstruct(
                scenario_result,
                time=hourly,
                depth=glider.depth.values,
                rho_reference=rho_reference,
                method="isopycnal_remap",
            )
        )
        exact_phase = phase_product.scenario_phase_offset.values[scenario_index]
        track_linear.append(
            pycnotide.reconstruct(
                scenario_result,
                time=glider.sample_time.values,
                depth=glider.depth.values,
                rho_reference=glider.density_reference.values,
                phase_offset=exact_phase,
                method="linear",
            )
        )
        track_remap.append(
            pycnotide.reconstruct(
                scenario_result,
                time=glider.sample_time.values,
                depth=glider.depth.values,
                rho_reference=glider.density_reference.values,
                phase_offset=exact_phase,
                method="isopycnal_remap",
            )
        )
    output = outputs / "guam2019_reconstruction.nc"
    xr.Dataset(
        {
            "reference_displacement": (
                ("scenario", "depth", "time"),
                np.stack([item.xi for item in reference_linear]),
            ),
            "reference_isopycnal_vertical_velocity": (
                ("scenario", "depth", "time"),
                np.stack([item.w_iso for item in reference_linear]),
            ),
            "reference_density_linear": (
                ("scenario", "depth", "time"),
                np.stack([item.rho for item in reference_linear]),
            ),
            "reference_density_remap": (
                ("scenario", "depth", "time"),
                np.stack([item.rho for item in reference_remap]),
            ),
            "reference_remap_support": (
                ("scenario", "depth", "time"),
                np.stack([item.mask for item in reference_remap]),
            ),
            "exact_track_displacement": (
                ("scenario", "depth", "profile"),
                np.stack([item.xi for item in track_linear]),
            ),
            "exact_track_isopycnal_vertical_velocity": (
                ("scenario", "depth", "profile"),
                np.stack([item.w_iso for item in track_linear]),
            ),
            "exact_track_density_linear": (
                ("scenario", "depth", "profile"),
                np.stack([item.rho for item in track_linear]),
            ),
            "exact_track_density_remap": (
                ("scenario", "depth", "profile"),
                np.stack([item.rho for item in track_remap]),
            ),
            "exact_track_remap_support": (
                ("scenario", "depth", "profile"),
                np.stack([item.mask for item in track_remap]),
            ),
            "density_reference": (("depth",), rho_reference),
        },
        coords={
            "scenario": list(scenario_names),
            "depth": glider.depth.values,
            "time": hourly,
            "profile": glider.profile.values,
            "profile_time": (("profile",), glider.profile_time.values),
        },
        attrs={
            "schema": "pycnotide_guam_reconstruction_v1",
            "phase_convention": result.phase_convention,
            "depth_positive": "down",
            "displacement_positive": "up",
            "reference_x": phase_product.attrs["reference_x"],
            "reference_y": phase_product.attrs["reference_y"],
            "exact_track_sampling": "sample-level time and underwater EPSG:32655 location",
        },
    ).to_netcdf(
        output,
        engine="netcdf4",
        encoding={
            "reference_density_remap": {"zlib": True},
            "exact_track_density_remap": {"zlib": True},
        },
    )
    return output


def render_report(config: dict, repository: Path) -> list[Path]:
    xr, pd, plt = _imports()
    processed, outputs = _paths(config, repository)
    figures = outputs / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    products: list[Path] = []
    with xr.open_dataset(processed / "guam_seaglider_level3_processed.nc") as opened:
        glider = opened.load()
    with xr.open_dataset(outputs / "guam2019_harmonics.nc") as opened:
        harmonics = opened.load()
    with xr.open_dataset(outputs / "guam2019_reconstruction.nc") as opened:
        reconstruction = opened.load()
    phase = pd.read_csv(outputs / "phase_fit_summary.csv")
    objective = pd.read_csv(outputs / "phase_objective_samples.csv")
    current = pd.read_csv(outputs / "current_sampling_comparison.csv")
    current_statistics = pd.read_csv(outputs / "current_statistics.csv")

    def save_pair(figure, stem: str) -> None:
        png = figures / f"{stem}.png"
        pdf = figures / f"{stem}.pdf"
        figure.savefig(png, dpi=300)
        figure.savefig(pdf)
        plt.close(figure)
        products.extend((png, pdf))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for name in np.unique(glider.glider.values):
        mask = glider.glider.values == name
        axes[0].scatter(
            glider.lon_dive.values[mask],
            glider.lat_dive.values[mask],
            s=6,
            label=str(name),
        )
    axes[0].set(xlabel="Longitude", ylabel="Latitude", title="Seaglider profile coverage")
    axes[0].legend()
    axes[1].hist(
        pd.to_datetime(glider.profile_time.values), bins=30, color="#2979b8"
    )
    axes[1].set(xlabel="Date", ylabel="Profiles", title="Temporal coverage")
    save_pair(fig, "figure01_sampling_coverage")

    reference = np.nanmedian(glider.density_reference.values, axis=0)
    gradient = np.nanmedian(glider.density_gradient.values, axis=0)
    displacement_rms = np.sqrt(np.nanmean(glider.displacement.values**2, axis=0))
    fig, axes = plt.subplots(1, 3, figsize=(12, 7), sharey=True, constrained_layout=True)
    axes[0].plot(reference, glider.depth, color="#222222")
    axes[1].plot(gradient, glider.depth, color="#288f68")
    axes[2].plot(displacement_rms, glider.depth, color="#b14c37")
    labels = (
        "Reference potential density (kg m$^{-3}$)",
        "Median $d\\rho_0/dd$ (kg m$^{-4}$)",
        "Observed displacement RMS (m)",
    )
    for axis, label in zip(axes, labels, strict=True):
        axis.invert_yaxis()
        axis.grid(alpha=0.25)
        axis.set(xlabel=label)
    axes[0].set(ylabel="Depth (m)")
    save_pair(fig, "figure02_density_reference_and_displacement")

    fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True, constrained_layout=True)
    for name in config["analysis"]["phase_constituents"]:
        selected = harmonics.sel(constituent=name)
        axes[0].plot(selected.amplitude_displacement, harmonics.depth, label=name)
        axes[1].plot(np.rad2deg(selected.phase_lag), harmonics.depth, label=name)
    for axis in axes:
        axis.invert_yaxis()
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set(xlabel="Displacement amplitude (m)", ylabel="Depth (m)")
    axes[1].set(xlabel="Phase lag (degree)")
    save_pair(fig, "figure03_harmonic_profiles")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for axis, component in zip(axes, ("u", "v"), strict=True):
        observed = current[f"glider_{component}_projected_m_s"]
        modeled = current[f"hycom_{component}_background_m_s"]
        axis.scatter(observed, modeled, s=9, alpha=0.55, color="#3b79a8")
        finite = np.isfinite(observed) & np.isfinite(modeled)
        if np.any(finite):
            lower = min(observed[finite].min(), modeled[finite].min())
            upper = max(observed[finite].max(), modeled[finite].max())
            axis.plot([lower, upper], [lower, upper], color="black", lw=1)
        axis.set(
            xlabel=f"Glider {component} (m s$^{{-1}}$)",
            ylabel=f"HYCOM {component} (m s$^{{-1}}$)",
            title=f"Projected {component}-component",
        )
        axis.grid(alpha=0.25)
    save_pair(fig, "figure04_hycom_glider_current_comparison")

    combinations = [
        (scenario, constituent)
        for scenario in ("zero_current", "hycom_current")
        for constituent in config["analysis"]["phase_constituents"]
    ]
    column_count = min(2, len(combinations))
    row_count = int(np.ceil(len(combinations) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6 * column_count, 4.5 * row_count),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, (scenario, constituent) in zip(
        flat_axes[: len(combinations)], combinations, strict=True
    ):
        selected = objective[
            (objective.scenario == scenario) & (objective.constituent == constituent)
        ]
        selected = selected[np.isfinite(selected.objective)]
        image = axis.scatter(
            selected.direction_deg,
            selected.phase_speed_m_s,
            c=selected.objective,
            s=13,
            cmap="viridis_r",
        )
        optimum = phase[
            (phase.scenario == scenario) & (phase.constituent == constituent)
        ].iloc[0]
        axis.scatter(
            optimum.direction_deg,
            optimum.phase_speed_m_s,
            marker="*",
            s=130,
            color="white",
            edgecolor="black",
        )
        axis.set(
            xlabel="Propagation bearing (degree)",
            ylabel="Effective phase speed (m s$^{-1}$)",
            title=f"{scenario.replace('_', ' ')}: {constituent}",
        )
        fig.colorbar(image, ax=axis, label="Variable-projection objective")
    for axis in flat_axes[len(combinations) :]:
        axis.set_visible(False)
    save_pair(fig, "figure05_bearing_speed_objective_samples")

    values = reconstruction.reference_displacement.values
    limit = float(np.nanpercentile(np.abs(values), 99.0))
    fig, axes = plt.subplots(
        1, len(reconstruction.scenario), figsize=(13, 6), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    image = None
    for index, axis in enumerate(axes):
        image = axis.pcolormesh(
            reconstruction.time,
            reconstruction.depth,
            values[index],
            shading="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.invert_yaxis()
        axis.set(
            xlabel="Time",
            ylabel="Depth (m)" if index == 0 else "",
            title=str(reconstruction.scenario.values[index]).replace("_", " "),
        )
    fig.colorbar(image, ax=axes, label="Displacement (m; positive up)")
    save_pair(fig, "figure06_reconstructed_displacement_sensitivity")

    table = phase.to_html(index=False, float_format=lambda value: f"{value:.5g}")
    current_table = current_statistics.to_html(
        index=False, float_format=lambda value: f"{value:.5g}"
    )
    links = "".join(
        f'<li><a href="figures/{item.name}">{item.name}</a></li>' for item in products
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>PycnoTide Guam 2019</title>
<style>body{{font:16px/1.5 system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:.35rem}}</style></head>
<body><h1>PycnoTide Guam 2019 demonstration</h1>
<p>Observed Level-3 density harmonics and straight-ray current sensitivity. The
HYCOM-current case is not bent-ray tracing or independent observational truth.</p>
<p>The reference fields use the supplied five-day/10 m T_ref and S_ref. Primary
support requires finite PD with good temperature and salinity flags. Confidence
intervals use complete-profile 48-hour block resampling. The fitted speed is an
effective sampling parameter rather than an identified vertical-mode speed.</p>
<h2>Figures</h2><ul>{links}</ul>
<h2>Phase fits and held-out skill</h2>{table}
<h2>HYCOM/glider current consistency</h2>
<p>This comparison is approximate because the glider flight-model current and the
HYCOM five-day, 0–1000 m background average have different sampling volumes.</p>
{current_table}
<h2>Interpretation limits</h2><p>Zero-current and HYCOM-current reconstructions use
identical observational support. Neither is preferred without identifiable objective
structure and improved held-out skill. Finite-amplitude remapping is kinematic and
does not add nonlinear internal-wave dynamics.</p></body></html>"""
    report_path = outputs / "index.html"
    report_path.write_text(report, encoding="utf-8")
    products.append(report_path)
    return products


def validate_products(config: dict, repository: Path) -> dict:
    xr, pd, _ = _imports()
    processed, outputs = _paths(config, repository)
    runs = (repository / config["paths"]["runs"]).resolve()
    required = [
        processed / "guam_seaglider_level3_processed.nc",
        processed / "hycom_background_current_epsg32655.nc",
        outputs / "guam2019_harmonics.nc",
        outputs / "guam2019_current_phase_sensitivity.nc",
        outputs / "guam2019_reconstruction.nc",
        outputs / "harmonic_profile_summary.csv",
        outputs / "window_harmonic_summary.csv",
        outputs / "phase_fit_summary.csv",
        outputs / "phase_objective_samples.csv",
        outputs / "current_sampling_comparison.csv",
        outputs / "current_statistics.csv",
        outputs / "index.html",
        runs / "podaac_inventory.json",
        runs / "source_manifest.json",
    ]
    figure_stems = [
        "figure01_sampling_coverage",
        "figure02_density_reference_and_displacement",
        "figure03_harmonic_profiles",
        "figure04_hycom_glider_current_comparison",
        "figure05_bearing_speed_objective_samples",
        "figure06_reconstructed_displacement_sensitivity",
    ]
    required.extend(
        outputs / "figures" / f"{stem}.{extension}"
        for stem in figure_stems
        for extension in ("png", "pdf")
    )
    missing = [path.name for path in required if not path.is_file() or path.stat().st_size == 0]
    checks: dict[str, bool | int | float | list] = {"missing": missing}
    if not missing:
        with xr.open_dataset(required[0]) as glider:
            checks["three_gliders"] = len(np.unique(glider.glider.values)) == 3
            checks["finite_displacement"] = bool(np.any(np.isfinite(glider.displacement.values)))
            checks["depth_0_1000"] = bool(
                np.min(glider.depth.values) >= 0 and np.max(glider.depth.values) <= 1000
            )
            accepted_time = glider.sample_time.values[
                glider.good_primary.values.astype(bool)
            ].astype("datetime64[ns]")
        with xr.open_dataset(required[1]) as current:
            checks["finite_current"] = bool(np.any(np.isfinite(current.u_background.values)))
            checks["monotonic_current_time"] = bool(
                np.all(
                    np.diff(current.time.values.astype("datetime64[ns]"))
                    > np.timedelta64(0, "s")
                )
            )
            current_time = current.time.values.astype("datetime64[ns]")
            checks["hycom_brackets_glider"] = bool(
                np.min(current_time) <= np.min(accepted_time)
                and np.max(current_time) >= np.max(accepted_time)
            )
        with xr.open_dataset(required[2]) as harmonics:
            checks["nonnegative_amplitude"] = bool(
                np.nanmin(harmonics.amplitude_displacement.values) >= 0
            )
            checks["phase_convention_present"] = "phase_convention" in harmonics.attrs
        with xr.open_dataset(required[3]) as sensitivity:
            shared = sensitivity.shared_observational_support.values.astype(bool)
            common_phase = np.all(
                np.isfinite(sensitivity.scenario_phase_offset.values), axis=(0, 1)
            )
            checks["identical_phase_support"] = bool(np.array_equal(shared, common_phase))
            checks["two_phase_scenarios"] = set(
                str(value) for value in sensitivity.scenario_name.values
            ) == {"zero_current", "hycom_current"}
        with xr.open_dataset(required[4]) as reconstruction:
            checks["reference_and_exact_track"] = all(
                name in reconstruction
                for name in (
                    "reference_displacement",
                    "reference_density_remap",
                    "exact_track_displacement",
                    "exact_track_density_remap",
                )
            )
            checks["reconstruction_scenarios"] = set(
                str(value) for value in reconstruction.scenario.values
            ) == {"zero_current", "hycom_current"}
        frame = pd.read_csv(outputs / "phase_fit_summary.csv")
        checks["both_current_cases"] = set(frame.scenario) == {
            "zero_current",
            "hycom_current",
        }
        checks["phase_fit_finite"] = bool(
            np.all(np.isfinite(frame[["direction_deg", "phase_speed_m_s", "objective"]]))
        )
        inventory = json.loads((runs / "podaac_inventory.json").read_text(encoding="utf-8"))
        raw_directory = (repository / config["paths"]["glider_raw"]).resolve()
        fingerprints = []
        for record in inventory["granules"]:
            path = raw_directory / record["filename"]
            fingerprints.append(
                path.is_file()
                and path.stat().st_size == int(record["size_bytes"])
                and file_checksum(path, record["checksum_algorithm"]).lower()
                == str(record["checksum"]).lower()
            )
        checks["three_cmr_fingerprints"] = len(fingerprints) == 3 and all(fingerprints)
        report_text = (outputs / "index.html").read_text(encoding="utf-8")
        checks["report_links_resolve"] = all(
            f"figures/{stem}.{extension}" in report_text
            for stem in figure_stems
            for extension in ("png", "pdf")
        )
    passed = not missing and all(bool(value) for key, value in checks.items() if key != "missing")
    report = {
        "schema": "pycnotide_guam_final_qa_v1",
        "status": "pass" if passed else "fail",
        "checks": checks,
        "artifacts": [
            {"name": path.name, "sha256": file_sha256(path)} for path in required if path.is_file()
        ],
    }
    write_json(outputs / "final_qa.json", report)
    if not passed:
        raise RuntimeError(f"Guam validation failed: {checks}")
    return report
