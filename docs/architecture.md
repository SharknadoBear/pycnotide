# Architecture

`src/pycnotide` is the portable numerical core. It contains no PO.DAAC, HYCOM,
SCHISM, catalog, or authentication code. `studies/guam_2019` owns campaign-specific
inventory, protected acquisition, NetCDF normalization, GSW conversion, figures,
and reports. Native and derived scientific data are ignored by Git.

The solver accepts arrays, builds an explicit nuisance-plus-harmonic design matrix,
solves it by weighted SVD, and returns immutable result containers. Phase models
only supply phase offsets; they do not alter astronomical frequencies or silently
modify the local regression. Reconstruction consumes the same coefficient convention.

The Guam adapter writes three layers of products: normalized observation/current
inputs, harmonic and phase-sensitivity diagnostics, and reference-location plus
exact-track reconstructions. Both current scenarios are evaluated on an explicit
shared observational-support mask. Optimizer evaluation samples are retained so the
bearing-speed identifiability surface is inspectable rather than reduced to one best
point.
