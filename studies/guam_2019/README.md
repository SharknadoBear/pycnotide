# Guam 2019 demonstration

This study uses three protected Level-3 PO.DAAC Seaglider granules and a bounded
GOFS 3.1 HYCOM current subset. Acquisition is outside the PycnoTide core.

Before protected download, rotate any credential that has been exposed and configure
the replacement Earthdata Login in the user account outside this repository. Set
`PYCNOTIDE_EARTHDATA_ROTATED=1` only after that rotation. No credential is accepted
through the JSON configuration or command line.

HYCOM acquisition uses the external `hycom-fetcher` contract. Set
`HYCOM_FETCHER_SCRIPT` to its `hycom_fetcher.py` entry point. The workflow inventories
the source, estimates the bounded request, checks storage, fetches the hash-bound plan,
and requires a passing health report before preprocessing.

On Windows, libnetcdf currently fails the HTTPS OPeNDAP handshake at the official
HYCOM TDS even though the HTTPS catalog and DAP metadata endpoints are healthy. The
configuration therefore uses the official `tds.hycom.org` OPeNDAP dataset over HTTP;
the inventory hash and hash-bound request preserve exact source identity.

```text
python studies/guam_2019/guam_internal_tide.py \
  --config studies/guam_2019/config/guam_2019.json \
  --mode preflight|inventory|download|preprocess|fit|reconstruct|render|validate|all
```

The current-aware result is a straight-ray, frozen-background eikonal sensitivity.
It is not bent-ray tracing, full modal dynamics, or a direct SCHISM validation.
Bearing and effective-speed optimization uses deterministic native levels spaced
10 m apart over 100--900 m so the 1 m Level-3 grid is not treated as 801 independent
vertical replicates. Selected observations retain their exact sample times and
underwater positions; accepted phase and reconstruction products are then evaluated
on every supported native depth level.
The report includes the supplied density reference, stratification/displacement QA,
harmonic uncertainty, an approximate glider/HYCOM current comparison, sampled
bearing-speed objective surfaces, held-out skill, and both reconstruction cases.
