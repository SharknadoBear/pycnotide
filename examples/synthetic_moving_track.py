"""Minimal moving-track M2 solve and reconstruction example."""

from __future__ import annotations

import numpy as np

import pycnotide


def main() -> None:
    rng = np.random.default_rng(7)
    depth = np.arange(0.0, 1000.1, 50.0)
    seconds = np.sort(rng.uniform(0.0, 20.0 * 86400.0, 300))
    time = np.datetime64("2019-10-03") + seconds.astype("timedelta64[s]")
    x = rng.uniform(-30_000.0, 30_000.0, seconds.size)
    y = rng.uniform(-20_000.0, 20_000.0, seconds.size)
    density_reference = 1024.0 + 0.002 * depth
    displacement_coefficient = 8.0 * np.exp(-((depth - 450.0) / 250.0) ** 2)
    phase = pycnotide.PlaneWavePhase(
        {"M2": -5.0e-5}, {"M2": 2.5e-5}, 0.0, 0.0
    ).offsets(("M2",), time, x, y)[0]
    angle = pycnotide.CATALOG["M2"].frequency_rad_s * seconds + phase
    density = density_reference[None, :] + 0.002 * np.cos(angle)[:, None] * (
        displacement_coefficient[None, :]
    )
    result = pycnotide.solve(
        time=time,
        depth=depth,
        rho=density,
        rho_reference=density_reference,
        x=x,
        y=y,
        constituents=("M2",),
        phase_offset=phase[None, :],
    )
    prediction = pycnotide.reconstruct(
        result,
        time=time[:48],
        depth=depth,
        rho_reference=density_reference,
        method="isopycnal_remap",
    )
    peak = np.nanmax(result.amplitude_displacement)
    print(f"Recovered peak displacement amplitude: {peak:.3f} m")
    print(f"Reconstruction shape: {prediction.rho.shape}")


if __name__ == "__main__":
    main()
