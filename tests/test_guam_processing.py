from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from studies.guam_2019.acquisition import (
    download_podaac,
    file_checksum,
    write_hycom_request,
)
from studies.guam_2019.analysis import (
    fit_harmonics,
    fit_phase_sensitivity,
    reconstruct_products,
    render_report,
)
from studies.guam_2019.guam_internal_tide import load_config, preflight
from studies.guam_2019.processing import (
    _profile_vector,
    preprocess_gliders,
    preprocess_hycom,
)

BASE_CONFIG = Path(__file__).parents[1] / "studies/guam_2019/config/guam_2019.json"


def _config(tmp_path: Path) -> dict:
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config["paths"] = {
        "glider_raw": "data/raw/glider",
        "hycom_raw": "data/raw/hycom.nc",
        "processed": "data/processed",
        "runs": "runs",
        "outputs": "outputs",
    }
    config["analysis"]["current_average_days"] = 1
    config["analysis"]["current_projected_grid_m"] = 20_000.0
    config["hycom"]["bbox"] = [144.0, 13.4, 144.4, 13.8]
    return config


def _write_glider(path: Path, glider: int) -> None:
    import gsw

    profile = np.arange(4)
    depth = np.arange(0.0, 1000.1, 100.0)
    shape = (profile.size, depth.size)
    longitude = 144.15 + 0.01 * profile[:, None] + np.zeros(shape)
    latitude = 13.55 + 0.005 * profile[:, None] + np.zeros(shape)
    pressure = np.broadcast_to(depth[None, :], shape)
    temperature = 28.0 - 0.018 * pressure
    salinity = 34.6 + 0.0012 * pressure
    absolute = gsw.SA_from_SP(salinity, pressure, longitude, latitude)
    conservative = gsw.CT_from_t(absolute, temperature, pressure)
    reference = gsw.sigma0(absolute, conservative) + 1000.0
    seconds = (
        np.arange(profile.size)[:, None] * 12 * 3600
        + np.arange(depth.size)[None, :] * 60
        + glider * 120
    )
    time = np.datetime64("2019-10-03") + seconds.astype("timedelta64[s]")
    flags = np.ones(shape)
    flags[0, 2] = 0
    dataset = xr.Dataset(
        {
            "PD": (("profile", "z"), reference + 0.02 * np.cos(seconds / 4000.0)),
            "T_ref": (("profile", "z"), temperature),
            "S_ref": (("profile", "z"), salinity),
            "P": (("profile", "z"), pressure),
            "lon": (("profile", "z"), longitude),
            "lat": (("profile", "z"), latitude),
            "time": (("profile", "z"), time),
            "T_flags": (("profile", "z"), flags),
            "S_flags": (("profile", "z"), flags),
            "dive": (("profile",), profile + 1),
            "u_dive": (("profile",), np.full(profile.size, 0.1)),
            "v_dive": (("profile",), np.full(profile.size, -0.05)),
            "lon_dive": (("profile",), np.nanmedian(longitude, axis=1)),
            "lat_dive": (("profile",), np.nanmedian(latitude, axis=1)),
        },
        coords={"profile": profile, "z": depth},
    )
    dataset.to_netcdf(path, engine="netcdf4")


def _write_hycom(path: Path) -> None:
    time = np.arange(
        np.datetime64("2019-10-01"),
        np.datetime64("2019-10-06"),
        np.timedelta64(3, "h"),
    )
    depth = np.asarray([0.0, 100.0, 500.0, 1000.0])
    latitude = np.asarray([13.4, 13.6, 13.8])
    longitude = np.asarray([144.0, 144.2, 144.4])
    shape = (time.size, depth.size, latitude.size, longitude.size)
    u = np.full(shape, 0.1, dtype=np.float32)
    v = np.full(shape, 0.2, dtype=np.float32)
    xr.Dataset(
        {
            "water_u": (("time", "depth", "lat", "lon"), u),
            "water_v": (("time", "depth", "lat", "lon"), v),
        },
        coords={
            "time": time,
            "depth": depth,
            "lat": latitude,
            "lon": longitude,
        },
    ).to_netcdf(path, engine="netcdf4")


def test_glider_and_hycom_preprocessing(tmp_path):
    config = _config(tmp_path)
    glider_dir = tmp_path / config["paths"]["glider_raw"]
    glider_dir.mkdir(parents=True)
    for number in (178, 179, 180):
        _write_glider(glider_dir / f"sg{number}_synthetic_L3.nc", number)
    hycom = tmp_path / config["paths"]["hycom_raw"]
    hycom.parent.mkdir(parents=True, exist_ok=True)
    _write_hycom(hycom)

    glider_output = preprocess_gliders(config, tmp_path)
    current_output = preprocess_hycom(config, tmp_path)
    with xr.open_dataset(glider_output) as glider:
        assert glider.sizes == {"profile": 12, "depth": 11}
        assert len(np.unique(glider.glider.values)) == 3
        assert np.isnan(glider.density.values[0, 2])
        assert np.isnan(glider.displacement.values[0, 2])
        assert np.isfinite(glider.density_all_finite.values[0, 2])
        assert glider.interpolated_or_flagged.values[0, 2]
        assert np.nanmedian(glider.density_reference.values) > 1000.0
        assert str(glider.attrs["depth_positive"]) == "down"
    with xr.open_dataset(current_output) as current:
        assert np.all(np.diff(current.time.values) > np.timedelta64(0, "s"))
        speed = np.hypot(current.u_background.values, current.v_background.values)
        np.testing.assert_allclose(
            speed[np.isfinite(speed)], np.hypot(0.1, 0.2), rtol=2e-5
        )
        assert current.attrs["vector_rotation"] == "east/north rotated to projected grid axes"


def test_full_dive_metadata_maps_to_descent_ascent_half_profiles():
    dataset = xr.Dataset(
        {
            "PD": (("profile", "depth"), np.ones((6, 2))),
            "u_dive": (("dive",), np.asarray([0.1, 0.2, 0.3])),
        }
    )
    np.testing.assert_allclose(
        _profile_vector(dataset, "u_dive", "profile", "depth"),
        [0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
    )
    invalid = dataset.assign(u_dive=(("other",), np.asarray([0.1, 0.2])))
    with pytest.raises(ValueError, match="cannot map"):
        _profile_vector(invalid, "u_dive", "profile", "depth")


def test_campaign_fit_and_reconstruction_netcdf_interfaces(tmp_path):
    config = _config(tmp_path)
    config["analysis"].update(
        {
            "constituents": ["M2"],
            "phase_constituents": ["M2"],
            "bootstrap_resamples": 2,
            "window_days": 1,
            "window_step_days": 1,
            "background": "linear_time",
            "direction_bounds_deg": [280.0, 310.0],
            "phase_speed_bounds_m_s": [1.5, 2.8],
        }
    )
    glider_dir = tmp_path / config["paths"]["glider_raw"]
    glider_dir.mkdir(parents=True)
    for number in (178, 179, 180):
        _write_glider(glider_dir / f"sg{number}_synthetic_L3.nc", number)
    hycom = tmp_path / config["paths"]["hycom_raw"]
    hycom.parent.mkdir(parents=True, exist_ok=True)
    _write_hycom(hycom)
    preprocess_gliders(config, tmp_path)
    preprocess_hycom(config, tmp_path)

    harmonic_path = fit_harmonics(config, tmp_path)
    phase_path = fit_phase_sensitivity(config, tmp_path)
    reconstruction_path = reconstruct_products(config, tmp_path)
    rendered = render_report(config, tmp_path)
    with xr.open_dataset(harmonic_path) as harmonics:
        assert harmonics.attrs["phase_convention"].startswith("C=A*exp")
        assert harmonics.attrs["bootstrap_resamples"] == 2
    with xr.open_dataset(phase_path) as phase:
        assert set(phase.scenario_name.values.astype(str)) == {
            "zero_current",
            "hycom_current",
        }
        shared = phase.shared_observational_support.values.astype(bool)
        actual = np.all(np.isfinite(phase.scenario_phase_offset.values), axis=(0, 1))
        np.testing.assert_array_equal(shared, actual)
    with xr.open_dataset(reconstruction_path) as reconstruction:
        assert reconstruction.reference_displacement.shape[0] == 2
        assert reconstruction.exact_track_displacement.shape[-1] == 12
        assert set(reconstruction.scenario.values.astype(str)) == {
            "zero_current",
            "hycom_current",
        }
    assert len(rendered) == 13
    assert all(path.is_file() and path.stat().st_size > 0 for path in rendered)


def test_request_and_protected_download_gate(tmp_path, monkeypatch):
    config = _config(tmp_path)
    request = write_hycom_request(config, tmp_path / "request.json")
    assert request["bbox"] == [144.0, 13.4, 144.4, 13.8]
    assert "?" not in request["source"]
    fixture = tmp_path / "checksum.bin"
    fixture.write_bytes(b"pycnotide")
    assert file_checksum(fixture, "MD5") == "1924ffcd3ffdb1dbf632bd8bc0e3eb75"
    monkeypatch.delenv("PYCNOTIDE_EARTHDATA_ROTATED", raising=False)
    with pytest.raises(RuntimeError, match="rotate"):
        download_podaac(config, tmp_path, {"granules": []})


def test_config_rejects_secret_fields_and_preflight_records_no_value(tmp_path, monkeypatch):
    config = _config(tmp_path)
    unsafe = copy.deepcopy(config)
    unsafe["credential"] = {"password": "forbidden-placeholder"}
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(unsafe), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        load_config(path)

    monkeypatch.delenv("PYCNOTIDE_EARTHDATA_ROTATED", raising=False)
    monkeypatch.delenv("HYCOM_FETCHER_SCRIPT", raising=False)
    report = preflight(config, tmp_path, tmp_path / "runs")
    serialized = json.dumps(report).lower()
    assert report["status"] == "blocked"
    assert "password" not in serialized
    assert "token" not in serialized
