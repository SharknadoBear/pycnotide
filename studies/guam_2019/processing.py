"""Normalize Guam Seaglider profiles and HYCOM currents into model-neutral arrays."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _imports():
    try:
        import gsw
        import xarray as xr
        from pyproj import Proj, Transformer
        from scipy.interpolate import RegularGridInterpolator
    except ImportError as exc:  # pragma: no cover - guarded by preflight
        raise RuntimeError("Install pycnotide[guam] for campaign preprocessing") from exc
    return xr, gsw, Proj, Transformer, RegularGridInterpolator


def _profile_depth_array(ds, name: str, profile_dim: str, depth_dim: str) -> np.ndarray:
    if name not in ds:
        raise KeyError(f"Required Seaglider variable {name!r} is absent")
    data = ds[name]
    if profile_dim not in data.dims or depth_dim not in data.dims:
        raise ValueError(f"{name} must use profile and depth dimensions")
    return np.asarray(data.transpose(profile_dim, depth_dim).values)


def _profile_vector(ds, name: str, profile_dim: str, depth_dim: str) -> np.ndarray:
    if name not in ds:
        return np.full(ds.sizes[profile_dim], np.nan)
    data = ds[name]
    if profile_dim not in data.dims:
        raise ValueError(f"{name} lacks the profile dimension")
    if depth_dim in data.dims:
        values = np.asarray(data.transpose(profile_dim, depth_dim).values, dtype=float)
        return np.nanmedian(values, axis=1)
    extra = [dim for dim in data.dims if dim != profile_dim]
    values = np.asarray(data.transpose(profile_dim, *extra).values)
    return values.reshape(ds.sizes[profile_dim], -1)[:, 0]


def _matrix_time_mean(values: np.ndarray) -> np.ndarray:
    time = np.asarray(values).astype("datetime64[ns]")
    result = np.full(time.shape[0], np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    integer = time.astype("int64")
    valid = ~np.isnat(time)
    for i in range(time.shape[0]):
        if np.any(valid[i]):
            result[i] = np.datetime64(int(np.median(integer[i, valid[i]])), "ns")
    return result


def _glider_id(path: Path) -> str:
    match = re.search(r"sg\s*(\d+)", path.name, flags=re.IGNORECASE)
    return f"SG{match.group(1)}" if match else path.stem


def preprocess_gliders(config: dict, repository: Path) -> Path:
    xr, gsw, _, Transformer, _ = _imports()
    raw = (repository / config["paths"]["glider_raw"]).resolve()
    files = sorted(
        path
        for path in raw.glob("*.nc")
        if "l3" in path.name.lower() or "level3" in path.name.lower()
    )
    expected = int(config["collection"]["expected_granules"])
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} Level-3 files, found {len(files)}")
    collections: list[dict] = []
    common_depth = None
    for path in files:
        with xr.open_dataset(path, decode_times=True, mask_and_scale=True) as source:
            ds = source.load()
        if "z" not in ds or ds["z"].ndim != 1:
            raise ValueError(f"{path.name}: z must be a one-dimensional coordinate")
        depth_dim = ds["z"].dims[0]
        candidates = [dim for dim in ds["PD"].dims if dim != depth_dim]
        if len(candidates) != 1:
            raise ValueError(f"{path.name}: unable to identify the profile dimension")
        profile_dim = candidates[0]
        depth = np.asarray(ds["z"].values, dtype=float)
        order = np.argsort(depth)
        depth = depth[order]
        if np.any(depth < 0) or np.any(np.diff(depth) <= 0):
            raise ValueError(f"{path.name}: depth must be unique and positive down")
        if common_depth is None:
            common_depth = depth
        elif not np.array_equal(common_depth, depth):
            raise ValueError("The three Level-3 files do not share an exact depth grid")

        def matrix(
            name,
            dataset=ds,
            profile_dimension=profile_dim,
            depth_dimension=depth_dim,
            depth_order=order,
        ):
            return _profile_depth_array(
                dataset, name, profile_dimension, depth_dimension
            )[:, depth_order]

        density = np.asarray(matrix("PD"), dtype=float)
        temperature_ref = np.asarray(matrix("T_ref"), dtype=float)
        salinity_ref = np.asarray(matrix("S_ref"), dtype=float)
        pressure = np.asarray(matrix("P"), dtype=float)
        longitude = np.asarray(matrix("lon"), dtype=float)
        latitude = np.asarray(matrix("lat"), dtype=float)
        time = np.asarray(matrix("time")).astype("datetime64[ns]")
        t_flags = np.asarray(matrix("T_flags"), dtype=float)
        s_flags = np.asarray(matrix("S_flags"), dtype=float)
        good = (
            np.isfinite(density)
            & np.isfinite(temperature_ref)
            & np.isfinite(salinity_ref)
            & np.isfinite(pressure)
            & np.isfinite(longitude)
            & np.isfinite(latitude)
            & (t_flags >= 1.0)
            & (s_flags >= 1.0)
        )
        absolute_salinity = gsw.SA_from_SP(
            salinity_ref, pressure, longitude, latitude
        )
        conservative_temperature = gsw.CT_from_t(
            absolute_salinity, temperature_ref, pressure
        )
        density_reference = gsw.sigma0(
            absolute_salinity, conservative_temperature
        ) + 1000.0
        gradient = np.gradient(density_reference, depth, axis=1, edge_order=1)
        threshold = float(config["analysis"]["min_density_gradient_kg_m4"])
        displacement = (density - density_reference) / gradient
        displacement[~good | ~np.isfinite(gradient) | (gradient < threshold)] = np.nan
        density[~good] = np.nan
        density_reference[~good] = np.nan
        transformer = Transformer.from_crs(4326, config["analysis"]["epsg"], always_xy=True)
        x, y = transformer.transform(longitude, latitude)
        collections.append(
            {
                "glider": np.full(ds.sizes[profile_dim], _glider_id(path)),
                "profile_index": np.arange(ds.sizes[profile_dim]),
                "dive": _profile_vector(ds, "dive", profile_dim, depth_dim),
                "profile_time": _matrix_time_mean(time),
                "time": time,
                "longitude": longitude,
                "latitude": latitude,
                "x": x,
                "y": y,
                "density": density,
                "density_reference": density_reference,
                "density_gradient": gradient,
                "displacement": displacement,
                "good": good,
                "t_flags": t_flags,
                "s_flags": s_flags,
                "u_dive": _profile_vector(ds, "u_dive", profile_dim, depth_dim),
                "v_dive": _profile_vector(ds, "v_dive", profile_dim, depth_dim),
                "lon_dive": _profile_vector(ds, "lon_dive", profile_dim, depth_dim),
                "lat_dive": _profile_vector(ds, "lat_dive", profile_dim, depth_dim),
                "source": np.full(ds.sizes[profile_dim], path.name),
            }
        )
    assert common_depth is not None
    keys = collections[0].keys()
    merged = {key: np.concatenate([item[key] for item in collections], axis=0) for key in keys}
    count = merged["density"].shape[0]
    profile_id = np.asarray(
        [
            f"{g}_{int(i):05d}"
            for g, i in zip(
                merged["glider"], merged["profile_index"], strict=True
            )
        ]
    )
    output_dir = (repository / config["paths"]["processed"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "guam_seaglider_level3_processed.nc"
    dataset = xr.Dataset(
        data_vars={
            "density": (("profile", "depth"), merged["density"]),
            "density_reference": (("profile", "depth"), merged["density_reference"]),
            "density_gradient": (("profile", "depth"), merged["density_gradient"]),
            "displacement": (("profile", "depth"), merged["displacement"]),
            "good_primary": (("profile", "depth"), merged["good"]),
            "temperature_flag": (("profile", "depth"), merged["t_flags"]),
            "salinity_flag": (("profile", "depth"), merged["s_flags"]),
            "sample_time": (("profile", "depth"), merged["time"]),
            "longitude": (("profile", "depth"), merged["longitude"]),
            "latitude": (("profile", "depth"), merged["latitude"]),
            "x": (("profile", "depth"), merged["x"]),
            "y": (("profile", "depth"), merged["y"]),
            "profile_time": (("profile",), merged["profile_time"]),
            "u_dive": (("profile",), merged["u_dive"]),
            "v_dive": (("profile",), merged["v_dive"]),
            "lon_dive": (("profile",), merged["lon_dive"]),
            "lat_dive": (("profile",), merged["lat_dive"]),
            "dive": (("profile",), merged["dive"]),
            "glider": (("profile",), merged["glider"].astype(str)),
            "source_file": (("profile",), merged["source"].astype(str)),
        },
        coords={"profile": profile_id.astype(str), "depth": common_depth},
        attrs={
            "schema": "pycnotide_guam_seaglider_processed_v1",
            "depth_positive": "down",
            "displacement_positive": "up",
            "crs": f"EPSG:{config['analysis']['epsg']}",
            "primary_qc": "finite PD/T_ref/S_ref/P/position and T_flags,S_flags >= 1",
            "profile_count": count,
        },
    )
    dataset.to_netcdf(output, engine="netcdf4")
    return output


def _coordinate_name(ds, candidates: tuple[str, ...]) -> str:
    lowered = {name.lower(): name for name in ds.variables}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise KeyError(f"No coordinate found among {candidates}")


def _layer_thickness(depth: np.ndarray, lower: float, upper: float) -> np.ndarray:
    mid = 0.5 * (depth[:-1] + depth[1:])
    edges = np.concatenate(([max(lower, 0.0)], mid, [upper]))
    left = np.maximum(edges[:-1], lower)
    right = np.minimum(edges[1:], upper)
    return np.maximum(0.0, right - left)


def preprocess_hycom(config: dict, repository: Path) -> Path:
    xr, _, Proj, Transformer, RegularGridInterpolator = _imports()
    source = (repository / config["paths"]["hycom_raw"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with xr.open_dataset(source, decode_times=True, mask_and_scale=True) as opened:
        ds = opened.load()
    time_name = _coordinate_name(ds, ("time", "MT"))
    depth_name = _coordinate_name(ds, ("depth", "Depth"))
    lat_name = _coordinate_name(ds, ("lat", "Latitude"))
    lon_name = _coordinate_name(ds, ("lon", "Longitude"))
    u_name = next(name for name in config["hycom"]["variables"] if "u" in name.lower())
    v_name = next(name for name in config["hycom"]["variables"] if "v" in name.lower())
    depth = np.asarray(ds[depth_name].values, dtype=float)
    selected = (depth >= config["hycom"]["depth"][0]) & (
        depth <= config["hycom"]["depth"][1]
    )
    if np.count_nonzero(selected) < 2:
        raise ValueError("HYCOM has insufficient depth support over 0-1000 m")
    depth = depth[selected]
    u = np.asarray(
        ds[u_name].transpose(time_name, depth_name, lat_name, lon_name).values[:, selected],
        dtype=float,
    )
    v = np.asarray(
        ds[v_name].transpose(time_name, depth_name, lat_name, lon_name).values[:, selected],
        dtype=float,
    )
    thickness = _layer_thickness(depth, *config["hycom"]["depth"])
    valid = np.isfinite(u) & np.isfinite(v)
    denominator = np.sum(valid * thickness[None, :, None, None], axis=1)
    total = float(np.sum(thickness))
    support = denominator / total
    u_mean = np.nansum(u * thickness[None, :, None, None], axis=1) / denominator
    v_mean = np.nansum(v * thickness[None, :, None, None], axis=1) / denominator
    minimum = float(config["analysis"]["current_vertical_support_fraction"])
    u_mean[support < minimum] = np.nan
    v_mean[support < minimum] = np.nan

    time = np.asarray(ds[time_name].values).astype("datetime64[ns]")
    delta_hours = float(np.median(np.diff(time) / np.timedelta64(1, "h")))
    window = int(round(24.0 * config["analysis"]["current_average_days"] / delta_hours))
    if window % 2 == 0:
        window += 1
    minimum_periods = int(np.ceil(0.8 * window))
    current = xr.Dataset(
        {
            "u": (("time", "latitude", "longitude"), u_mean),
            "v": (("time", "latitude", "longitude"), v_mean),
            "vertical_support": (("time", "latitude", "longitude"), support),
        },
        coords={
            "time": time,
            "latitude": np.asarray(ds[lat_name].values, dtype=float),
            "longitude": np.asarray(ds[lon_name].values, dtype=float),
        },
    ).sortby(["latitude", "longitude"])
    current["u_background"] = current["u"].rolling(
        time=window, center=True, min_periods=minimum_periods
    ).mean()
    current["v_background"] = current["v"].rolling(
        time=window, center=True, min_periods=minimum_periods
    ).mean()

    transformer = Transformer.from_crs(4326, config["analysis"]["epsg"], always_xy=True)
    inverse = Transformer.from_crs(config["analysis"]["epsg"], 4326, always_xy=True)
    bbox = config["hycom"]["bbox"]
    corner_x, corner_y = transformer.transform(
        [bbox[0], bbox[0], bbox[2], bbox[2]], [bbox[1], bbox[3], bbox[1], bbox[3]]
    )
    spacing = float(config["analysis"]["current_projected_grid_m"])
    x_grid = np.arange(
        np.floor(min(corner_x) / spacing) * spacing,
        max(corner_x) + spacing,
        spacing,
    )
    y_grid = np.arange(
        np.floor(min(corner_y) / spacing) * spacing,
        max(corner_y) + spacing,
        spacing,
    )
    xx, yy = np.meshgrid(x_grid, y_grid)
    target_lon, target_lat = inverse.transform(xx, yy)
    points = np.column_stack([target_lat.ravel(), target_lon.ravel()])
    projected_u = np.full((time.size, y_grid.size, x_grid.size), np.nan, dtype=np.float32)
    projected_v = np.full_like(projected_u, np.nan)
    lat = np.asarray(current.latitude.values, dtype=float)
    lon = np.asarray(current.longitude.values, dtype=float)
    projection = Proj(f"EPSG:{config['analysis']['epsg']}")
    convergence = np.deg2rad(projection.get_factors(target_lon, target_lat).meridian_convergence)
    cosine, sine = np.cos(convergence), np.sin(convergence)
    for it in range(time.size):
        east = RegularGridInterpolator(
            (lat, lon), current.u_background.values[it], bounds_error=False, fill_value=np.nan
        )(points).reshape(xx.shape)
        north = RegularGridInterpolator(
            (lat, lon), current.v_background.values[it], bounds_error=False, fill_value=np.nan
        )(points).reshape(xx.shape)
        projected_u[it] = east * cosine - north * sine
        projected_v[it] = east * sine + north * cosine
    output_dir = (repository / config["paths"]["processed"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "hycom_background_current_epsg32655.nc"
    xr.Dataset(
        {
            "u_background": (("time", "y", "x"), projected_u),
            "v_background": (("time", "y", "x"), projected_v),
        },
        coords={"time": time, "x": x_grid, "y": y_grid},
        attrs={
            "schema": "pycnotide_hycom_background_current_v1",
            "source_variables": f"{u_name},{v_name}",
            "vertical_average_m": "0-1000",
            "minimum_vertical_support_fraction": minimum,
            "background_average_days": config["analysis"]["current_average_days"],
            "crs": f"EPSG:{config['analysis']['epsg']}",
            "vector_rotation": "east/north rotated to projected grid axes",
        },
    ).to_netcdf(output, engine="netcdf4")
    return output
