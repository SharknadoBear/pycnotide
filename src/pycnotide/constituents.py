"""Astronomical constituent definitions used by the compact solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Constituent:
    """A fixed astronomical constituent frequency."""

    name: str
    cycles_per_day: float

    @property
    def frequency_rad_s(self) -> float:
        return float(2.0 * np.pi * self.cycles_per_day / 86_400.0)


CATALOG: dict[str, Constituent] = {
    "Q1": Constituent("Q1", 0.893244),
    "O1": Constituent("O1", 0.929536),
    "P1": Constituent("P1", 0.997262),
    "K1": Constituent("K1", 1.002738),
    "N2": Constituent("N2", 1.895982),
    "M2": Constituent("M2", 1.932274),
    "S2": Constituent("S2", 2.0),
    "K2": Constituent("K2", 2.005476),
}


def get_constituents(names: tuple[str, ...] | list[str]) -> tuple[Constituent, ...]:
    """Return validated catalog entries in caller order."""

    result: list[Constituent] = []
    for raw in names:
        name = str(raw).upper()
        if name not in CATALOG:
            raise ValueError(f"Unknown constituent {raw!r}; choose from {sorted(CATALOG)}")
        result.append(CATALOG[name])
    if len({item.name for item in result}) != len(result):
        raise ValueError("Constituent names must be unique")
    return tuple(result)

