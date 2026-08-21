# PycnoTide

**PycnoTide** performs harmonic analysis and reconstruction of coherent internal tides
from moving hydrographic profiles. The core is array based and depends only on NumPy
and SciPy. Data acquisition and model-specific preprocessing remain outside the core.

The canonical coefficient and prediction convention is

\[
C=Ae^{-i\phi},\qquad q(t)=\Re\{C e^{i[\omega(t-t_0)+\psi]}\}.
\]

Depth is positive downward and isopycnal displacement is positive upward. Spatial
phase is negative along the propagation direction.

## Quick start

```python
import pycnotide

result = pycnotide.solve(
    time=time,
    depth=depth,
    rho=density,
    rho_reference=reference_density,
    constituents=("M2", "K1"),
)

prediction = pycnotide.reconstruct(result, time=time_out, depth=depth)
```

`reconstruct` also accepts exact-track timestamp matrices shaped
`(observation, depth)` and depth-dependent phase offsets. Unsupported paths stay
missing throughout linear and isopycnal-remap products.

The Guam campaign workflow is under `studies/guam_2019`. Raw PO.DAAC and HYCOM
files are never committed. The current-aware demonstration is a straight-ray eikonal
sensitivity, not bent-ray tracing or proof of a resolution-only/model-only cause.

## Development

```text
python -m pip install -e ".[test,guam]"
pytest
ruff check .
```
