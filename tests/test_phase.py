import numpy as np

import pycnotide


def circular_error_degrees(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_plane_wave_sign():
    phase = pycnotide.PlaneWavePhase({"M2": 2e-5}, {"M2": -1e-5}, 0.0, 0.0)
    offsets = phase.offsets(
        ("M2",), np.arange(2), np.array([1000.0, 2000.0]), np.array([500.0, 1000.0])
    )
    np.testing.assert_allclose(offsets[0], [-0.015, -0.03])


def test_zero_and_uniform_current_wavenumber_roots():
    omega = pycnotide.CATALOG["M2"].frequency_rad_s
    f = float(pycnotide.coriolis_parameter(14.0))
    speed = 2.0
    k0 = np.sqrt(omega**2 - f**2) / speed
    np.testing.assert_allclose(
        pycnotide.current_aware_wavenumber(omega, f, speed, 0.0), k0, rtol=1e-13
    )
    u = 0.25
    k = float(pycnotide.current_aware_wavenumber(omega, f, speed, u))
    sigma = omega - k * u
    np.testing.assert_allclose(sigma**2, f**2 + speed**2 * k**2, rtol=1e-12)
    assert k > 0 and sigma > abs(f)


def test_straight_ray_zero_current_phase_and_barrier():
    omega = pycnotide.CATALOG["M2"].frequency_rad_s
    f = float(pycnotide.coriolis_parameter(14.0))
    speed = 2.2
    model = pycnotide.StraightRayEikonalPhase(
        {"M2": 90.0}, {"M2": speed}, (0.0, 0.0), f, integration_step_m=2000.0
    )
    x = np.array([0.0, 10_000.0, -5_000.0])
    y = np.zeros(3)
    time = np.array([np.datetime64("2019-10-03")] * 3)
    phase = model.offsets(("M2",), time, x, y, np.datetime64("2019-10-03"))[0]
    k0 = np.sqrt(omega**2 - f**2) / speed
    np.testing.assert_allclose(phase, -k0 * x, atol=1e-12)

    grid = pycnotide.RegularGridCurrent(
        np.array([0.0, 3600.0]),
        np.array([-1000.0, 1000.0]),
        np.array([0.0, 10_000.0]),
        np.zeros((2, 2, 2)),
        np.zeros((2, 2, 2)),
    )
    current_model = pycnotide.StraightRayEikonalPhase(
        {"M2": 90.0},
        {"M2": speed},
        (0.0, 0.0),
        f,
        current=grid,
        current_reference_time=np.datetime64("2019-10-03"),
    )
    outside = current_model.offsets(
        ("M2",), time[:1], np.array([20_000.0]), np.array([0.0]), np.datetime64("2019-10-03")
    )
    assert np.isnan(outside[0, 0])


def test_uniform_and_piecewise_current_path_integrals():
    omega = pycnotide.CATALOG["M2"].frequency_rad_s
    f = float(pycnotide.coriolis_parameter(14.0))
    speed = 2.0
    time = np.array([np.datetime64("2019-10-03")])
    uniform = pycnotide.RegularGridCurrent(
        np.array([0.0, 3600.0]),
        np.array([-1000.0, 1000.0]),
        np.array([0.0, 8000.0]),
        np.full((2, 2, 2), 0.2),
        np.zeros((2, 2, 2)),
    )
    model = pycnotide.StraightRayEikonalPhase(
        {"M2": 90.0},
        {"M2": speed},
        (0.0, 0.0),
        f,
        current=uniform,
        current_reference_time=np.datetime64("2019-10-03"),
        integration_step_m=2000.0,
    )
    phase = model.offsets(
        ("M2",), time, np.array([8000.0]), np.array([0.0]), time[0]
    )[0, 0]
    k = float(pycnotide.current_aware_wavenumber(omega, f, speed, 0.2))
    np.testing.assert_allclose(phase, -k * 8000.0, rtol=1e-13)

    class PiecewiseCurrent:
        def sample(self, time_seconds, x, y):
            del time_seconds, y
            return np.where(np.asarray(x) < 4000.0, 0.2, -0.1), np.zeros_like(x)

    piecewise = pycnotide.StraightRayEikonalPhase(
        {"M2": 90.0},
        {"M2": speed},
        (0.0, 0.0),
        f,
        current=PiecewiseCurrent(),
        current_reference_time=np.datetime64("2019-10-03"),
        integration_step_m=2000.0,
    )
    result = piecewise.offsets(
        ("M2",), time, np.array([8000.0]), np.array([0.0]), time[0]
    )[0, 0]
    nodes = np.asarray([0.0, 2000.0, 4000.0, 6000.0, 8000.0])
    velocity = np.where(nodes < 4000.0, 0.2, -0.1)
    node_k = pycnotide.current_aware_wavenumber(omega, f, speed, velocity)
    expected = -np.trapezoid(node_k, nodes)
    np.testing.assert_allclose(result, expected, rtol=1e-13)
    assert np.isnan(pycnotide.current_aware_wavenumber(omega, f, speed, -5.0))


def test_variable_projection_recovers_zero_current_wave():
    rng = np.random.default_rng(8)
    n = 90
    seconds = np.sort(rng.uniform(0.0, 20.0 * 86400.0, n))
    time = np.datetime64("2019-10-03") + seconds.astype("timedelta64[s]")
    x = rng.uniform(-35_000.0, 35_000.0, n)
    y = rng.uniform(-25_000.0, 25_000.0, n)
    direction = 296.0
    speed = 2.1
    f = float(pycnotide.coriolis_parameter(14.0))
    phase_model = pycnotide.StraightRayEikonalPhase(
        {"M2": direction}, {"M2": speed}, (0.0, 0.0), f
    )
    phase = phase_model.offsets(("M2",), time, x, y, np.datetime64("2019-10-03"))[0]
    omega = pycnotide.CATALOG["M2"].frequency_rad_s
    values = np.cos(omega * seconds + phase)
    fit = pycnotide.fit_straight_ray_phase(
        time=time,
        x=x,
        y=y,
        values=values,
        constituent="M2",
        reference_time=np.datetime64("2019-10-03"),
        reference_position=(0.0, 0.0),
        coriolis_rad_s=f,
        direction_bounds=(270.0, 320.0),
        phase_speed_bounds=(1.5, 2.7),
        seed=11,
    )
    assert fit.success
    assert circular_error_degrees(fit.direction_deg, direction) < 1.0
    assert abs(fit.phase_speed_m_s - speed) < 0.05
    assert len(fit.metadata["objective_sample_value"]) > 20


def test_variable_projection_recovers_uniform_current_wave():
    rng = np.random.default_rng(19)
    n = 70
    seconds = np.sort(rng.uniform(0.0, 12.0 * 86400.0, n))
    reference_time = np.datetime64("2019-10-03")
    time = reference_time + seconds.astype("timedelta64[s]")
    x = rng.uniform(-20_000.0, 20_000.0, n)
    y = rng.uniform(-15_000.0, 15_000.0, n)
    direction = 302.0
    speed = 2.25
    f = float(pycnotide.coriolis_parameter(14.0))
    current = pycnotide.RegularGridCurrent(
        np.array([0.0, 12.0 * 86400.0]),
        np.array([-50_000.0, 50_000.0]),
        np.array([-50_000.0, 50_000.0]),
        np.full((2, 2, 2), 0.12),
        np.full((2, 2, 2), -0.04),
    )
    model = pycnotide.StraightRayEikonalPhase(
        {"M2": direction},
        {"M2": speed},
        (0.0, 0.0),
        f,
        current=current,
        current_reference_time=reference_time,
    )
    phase = model.offsets(("M2",), time, x, y, reference_time)[0]
    values = np.cos(pycnotide.CATALOG["M2"].frequency_rad_s * seconds + phase)
    fit = pycnotide.fit_straight_ray_phase(
        time=time,
        x=x,
        y=y,
        values=values,
        constituent="M2",
        reference_time=reference_time,
        reference_position=(0.0, 0.0),
        coriolis_rad_s=f,
        current=current,
        current_reference_time=reference_time,
        direction_bounds=(285.0, 315.0),
        phase_speed_bounds=(1.8, 2.7),
        seed=27,
    )
    assert fit.success
    assert circular_error_degrees(fit.direction_deg, direction) < 1.0
    assert abs(fit.phase_speed_m_s - speed) < 0.05
