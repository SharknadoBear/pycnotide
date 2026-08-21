"""Command-line entry point for the Guam 2019 PycnoTide demonstration."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from studies.guam_2019.acquisition import (  # type: ignore[import-not-found]
        cmr_inventory,
        download_podaac,
        file_sha256,
        run_hycom_fetch,
        run_hycom_inventory,
        write_hycom_request,
        write_json,
    )
    from studies.guam_2019.analysis import (  # type: ignore[import-not-found]
        fit_harmonics,
        fit_phase_sensitivity,
        reconstruct_products,
        render_report,
        validate_products,
    )
    from studies.guam_2019.processing import (  # type: ignore[import-not-found]
        preprocess_gliders,
        preprocess_hycom,
    )
else:  # pragma: no cover
    from .acquisition import (
        cmr_inventory,
        download_podaac,
        file_sha256,
        run_hycom_fetch,
        run_hycom_inventory,
        write_hycom_request,
        write_json,
    )
    from .analysis import (
        fit_harmonics,
        fit_phase_sensitivity,
        reconstruct_products,
        render_report,
        validate_products,
    )
    from .processing import preprocess_gliders, preprocess_hycom

MODES = (
    "preflight",
    "inventory",
    "download",
    "preprocess",
    "fit",
    "reconstruct",
    "render",
    "validate",
    "all",
)


def load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "pycnotide_guam_2019_v1":
        raise ValueError("Unsupported or absent Guam configuration schema")
    serialized = json.dumps(payload).lower()
    if any(word in serialized for word in ('"password"', '"token"', '"secret"')):
        raise ValueError("Credentials and secret fields are forbidden in campaign configuration")
    return payload


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def preflight(config: dict, repository: Path, run_dir: Path) -> dict:
    required = ("numpy", "scipy", "xarray", "netCDF4", "gsw", "pyproj", "pandas", "matplotlib")
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in required:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    request = write_hycom_request(config, run_dir / "hycom_request.json")
    bbox = request["bbox"]
    checks = {
        "python_at_least_3_11": sys.version_info >= (3, 11),
        "dependencies_present": not missing,
        "hycom_request_bounded": (
            len(bbox) == 4
            and bbox[0] < bbox[2]
            and bbox[1] < bbox[3]
            and request["start"] < request["end"]
        ),
        "no_query_string_in_hycom_source": "?" not in request["source"],
        "earthdata_rotation_gate_set": os.environ.get("PYCNOTIDE_EARTHDATA_ROTATED") == "1",
        "hycom_fetcher_configured": bool(os.environ.get("HYCOM_FETCHER_SCRIPT")),
    }
    report = {
        "schema": "pycnotide_guam_preflight_v1",
        "status": "ready" if all(checks.values()) else "blocked",
        "checks": checks,
        "missing_dependencies": missing,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": versions,
        },
        "notes": [
            "No credential value was inspected or recorded.",
            "The Earthdata rotation gate reports state only.",
        ],
    }
    write_json(run_dir / "preflight.json", report)
    return report


def inventory(config: dict, repository: Path, run_dir: Path) -> dict:
    del repository
    podaac = cmr_inventory(config, run_dir / "podaac_inventory.json")
    free = shutil.disk_usage(run_dir).free
    storage_ok = free > 4 * int(podaac["total_size_bytes"])
    hycom_inventory = run_hycom_inventory(config, run_dir)
    report = {
        "schema": "pycnotide_guam_inventory_summary_v1",
        "podaac_level3_granules": len(podaac["granules"]),
        "podaac_total_size_bytes": podaac["total_size_bytes"],
        "four_times_podaac_storage_gate": storage_ok,
        "hycom_inventory": hycom_inventory.name,
    }
    write_json(run_dir / "inventory_summary.json", report)
    if not storage_ok:
        raise RuntimeError("Local free space failed the four-times PO.DAAC storage gate")
    return report


def download(config: dict, repository: Path, run_dir: Path) -> list[Path]:
    inventory_path = run_dir / "podaac_inventory.json"
    if not inventory_path.is_file():
        cmr_inventory(config, inventory_path)
    podaac = json.loads(inventory_path.read_text(encoding="utf-8"))
    gliders = download_podaac(config, repository, podaac)
    hycom = run_hycom_fetch(config, repository, run_dir / "hycom_run")
    sources = gliders + [hycom]
    write_json(
        run_dir / "source_manifest.json",
        {
            "schema": "pycnotide_guam_sources_v1",
            "sources": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
                for path in sources
            ],
        },
    )
    return sources


def preprocess(config: dict, repository: Path, run_dir: Path) -> list[Path]:
    outputs = [preprocess_gliders(config, repository), preprocess_hycom(config, repository)]
    write_json(
        run_dir / "processing_manifest.json",
        {
            "schema": "pycnotide_guam_processing_v1",
            "outputs": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
                for path in outputs
            ],
            "depth_positive": "down",
            "displacement_positive": "up",
            "current_background_days": config["analysis"]["current_average_days"],
        },
    )
    return outputs


def fit(config: dict, repository: Path, run_dir: Path) -> list[Path]:
    outputs = [fit_harmonics(config, repository), fit_phase_sensitivity(config, repository)]
    write_json(
        run_dir / "fitting_manifest.json",
        {
            "schema": "pycnotide_guam_fitting_v1",
            "outputs": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
                for path in outputs
            ],
            "constituents": config["analysis"]["constituents"],
            "phase_constituents": config["analysis"]["phase_constituents"],
            "bootstrap_resamples": config["analysis"]["bootstrap_resamples"],
            "qualification": "straight-ray sensitivity; not bent-ray tracing",
        },
    )
    return outputs


def render(config: dict, repository: Path, run_dir: Path) -> list[Path]:
    products = render_report(config, repository)
    write_json(
        run_dir / "figure_manifest.json",
        {
            "schema": "pycnotide_guam_figures_v1",
            "products": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
                for path in products
            ],
            "png_dpi": 300,
        },
    )
    return products


def execute(mode: str, config: dict, repository: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if mode in {"preflight", "all"}:
        report = preflight(config, repository, run_dir)
        if mode == "all" and report["status"] != "ready":
            raise RuntimeError(f"Preflight blocked: {report['checks']}")
    if mode in {"inventory", "all"}:
        inventory(config, repository, run_dir)
    if mode in {"download", "all"}:
        download(config, repository, run_dir)
    if mode in {"preprocess", "all"}:
        preprocess(config, repository, run_dir)
    if mode in {"fit", "all"}:
        fit(config, repository, run_dir)
    if mode in {"reconstruct", "all"}:
        reconstruct_products(config, repository)
    if mode in {"render", "all"}:
        render(config, repository, run_dir)
    if mode in {"validate", "all"}:
        validate_products(config, repository)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    arguments = parser.parse_args()
    repository = repository_root()
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = repository / config_path
    config = load_config(config_path.resolve())
    run_dir = (repository / config["paths"]["runs"]).resolve()
    execute(arguments.mode, config, repository, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
