# Phase conventions

- All frequencies are positive and expressed in radians per second.
- Time is evaluated relative to an explicit `datetime64[ns]` reference epoch.
- `C=A exp(-i phi)`; reported phase lag is `phi=-arg(C)`.
- A plane wave traveling along positive `k` has `psi=-k dot (x-x0)`.
- Bearings are degrees clockwise from true/projected north.
- Depth is positive downward; displacement is positive upward.
- A positive density anomaly in stable positive-down stratification corresponds to
  positive upward displacement.

Every serialized result carries the convention string and reference time.

