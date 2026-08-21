import numpy as np

import pycnotide


def fitted_result():
    seconds = np.arange(0.0, 20.0 * 86400.0, 3600.0)
    time = np.datetime64("2019-10-03") + seconds.astype("timedelta64[s]")
    depth = np.arange(0.0, 1001.0, 50.0)
    rho0 = 1024.0 + 0.002 * depth
    omega = pycnotide.CATALOG["M2"].frequency_rad_s
    xi = 8.0 * np.exp(-((depth - 400.0) / 250.0) ** 2)
    anomaly = np.cos(omega * seconds)[:, None] * (xi * 0.002)[None, :]
    return pycnotide.solve(
        time=time,
        depth=depth,
        rho=rho0[None, :] + anomaly,
        rho_reference=rho0,
        constituents=("M2",),
        background="linear_time",
    ), time, rho0


def test_linear_reconstruction_and_velocity():
    result, time, rho0 = fitted_result()
    pred = pycnotide.reconstruct(result, time=time[:24], rho_reference=rho0, method="linear")
    assert pred.xi.shape == (rho0.size, 24)
    np.testing.assert_allclose(pred.rho - rho0[:, None], pred.rho_anomaly, atol=1e-12)
    assert np.nanmax(abs(pred.w_iso)) > 0


def test_remap_preserves_monotonic_density():
    result, time, rho0 = fitted_result()
    pred = pycnotide.reconstruct(
        result, time=time[:24], rho_reference=rho0, method="isopycnal_remap"
    )
    for column in pred.rho.T:
        finite = np.isfinite(column)
        assert np.all(np.diff(column[finite]) >= 0)


def test_exact_track_time_and_depth_phase_matrix():
    result, time, rho0 = fitted_result()
    track_time = np.broadcast_to(time[:5, None], (5, rho0.size))
    phase = np.zeros((1, 5, rho0.size))
    phase[0] = np.linspace(-0.3, 0.4, rho0.size)[None, :]
    prediction = pycnotide.reconstruct(
        result,
        time=track_time,
        rho_reference=np.broadcast_to(rho0[None, :], track_time.shape),
        phase_offset=phase,
        method="linear",
    )
    seconds = (track_time - result.reference_time) / np.timedelta64(1, "s")
    expected = np.real(
        result.complex_displacement[0][None, :]
        * np.exp(1j * (result.frequency_rad_s[0] * seconds + phase[0]))
    )
    assert prediction.xi.shape == (rho0.size, 5)
    np.testing.assert_allclose(prediction.xi, expected.T, atol=1e-12)
    assert prediction.metadata["exact_track"] is True
