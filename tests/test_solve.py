import numpy as np
import pytest

import pycnotide


def test_exact_irregular_multiconstituent_profile_recovery():
    rng = np.random.default_rng(4)
    n = 500
    depth = np.array([100.0, 300.0, 600.0])
    seconds = np.sort(rng.integers(0, int(35.0 * 86400.0), n)).astype(float)
    time = np.datetime64("2019-10-03") + seconds.astype("timedelta64[s]")
    x = rng.uniform(-30_000.0, 30_000.0, n)
    y = rng.uniform(-20_000.0, 20_000.0, n)
    reference = 1024.0 + 0.002 * depth
    coefficients = np.array(
        [
            [0.020 - 0.010j, 0.030 + 0.012j, 0.015 - 0.004j],
            [0.008 + 0.003j, 0.010 - 0.002j, 0.006 + 0.001j],
        ]
    )
    anomaly = np.zeros((n, depth.size))
    for ic, name in enumerate(("M2", "K1")):
        omega = pycnotide.CATALOG[name].frequency_rad_s
        anomaly += np.real(coefficients[ic, :, None] * np.exp(1j * omega * seconds)).T
    anomaly += (2.0e-8 * x + 1.0e-8 * y + 1.0e-8 * seconds)[:, None]
    result = pycnotide.solve(
        time=time,
        depth=depth,
        rho=reference[None, :] + anomaly,
        rho_reference=reference,
        x=x,
        y=y,
        constituents=("M2", "K1"),
        background="linear_space_time",
        reference_time=np.datetime64("2019-10-03"),
    )
    np.testing.assert_allclose(result.complex_density, coefficients, atol=2.0e-11)
    np.testing.assert_allclose(
        result.complex_displacement, coefficients / 0.002, atol=2.0e-8
    )
    assert np.all(result.variance_explained > 0.999999)
    assert result.phase_convention == pycnotide.PHASE_CONVENTION


def test_short_record_and_weak_gradient_warnings():
    time = np.datetime64("2020-01-01") + np.arange(20).astype("timedelta64[h]")
    depth = np.array([0.0, 10.0])
    rho0 = np.array([1025.0, 1025.00001])
    rho = np.broadcast_to(rho0, (time.size, 2)).copy()
    with pytest.warns(RuntimeWarning):
        result = pycnotide.solve(
            time=time,
            depth=depth,
            rho=rho,
            rho_reference=rho0,
            constituents=("M2", "N2"),
            background="linear_time",
        )
    assert "WEAK_STRATIFICATION" in result.warnings
    assert "UNRESOLVED_CONSTITUENTS" in result.warnings


def test_group_weights_and_nan_support():
    t = np.arange(200) * 1800.0
    omega = pycnotide.CATALOG["M2"].frequency_rad_s
    values = np.cos(omega * t)[:, None]
    values[::9] = np.nan
    result = pycnotide.solve(
        time=t,
        depth=np.array([100.0]),
        rho=values,
        constituents=("M2",),
        weights="group",
        group=np.repeat(np.arange(20), 10),
        background="linear_time",
        displacement=False,
    )
    np.testing.assert_allclose(result.complex_density[0, 0], 1.0 + 0.0j, atol=1e-12)
    assert result.valid_count[0] == np.count_nonzero(np.isfinite(values))
