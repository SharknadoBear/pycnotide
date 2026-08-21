"""Protected PO.DAAC and bounded HYCOM acquisition orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_checksum(path: Path, algorithm: str) -> str:
    """Return a CMR-declared checksum without accepting arbitrary algorithms."""

    normalized = algorithm.lower().replace("-", "")
    if normalized not in {"md5", "sha256"}:
        raise ValueError(f"Unsupported CMR checksum algorithm: {algorithm!r}")
    digest = hashlib.new(normalized)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmr_inventory(config: dict, output: Path) -> dict:
    concept = config["collection"]["concept_id"]
    url = f"{CMR_GRANULES}?collection_concept_id={concept}&page_size=100"
    request = urllib.request.Request(
        url, headers={"Accept": "application/vnd.nasa.cmr.umm_results+json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        payload = json.load(response)
    records: list[dict] = []
    for item in payload.get("items", []):
        umm = item["umm"]
        granule = str(umm.get("GranuleUR", ""))
        is_level3 = "l3" in granule.lower() or "level3" in granule.lower()
        if not is_level3:
            continue
        distributions = umm.get("DataGranule", {}).get(
            "ArchiveAndDistributionInformation", []
        )
        netcdf = next(
            (entry for entry in distributions if str(entry.get("Name", "")).endswith(".nc")),
            None,
        )
        if netcdf is None:
            continue
        temporal = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
        records.append(
            {
                "granule_ur": granule,
                "filename": netcdf["Name"],
                "size_bytes": int(netcdf.get("SizeInBytes", 0)),
                "checksum_algorithm": netcdf.get("Checksum", {}).get("Algorithm"),
                "checksum": netcdf.get("Checksum", {}).get("Value"),
                "begin": temporal.get("BeginningDateTime"),
                "end": temporal.get("EndingDateTime"),
            }
        )
    records.sort(key=lambda record: record["filename"])
    expected = int(config["collection"]["expected_granules"])
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} Level-3 granules, CMR returned {len(records)}")
    result = {
        "schema": "pycnotide_podaac_inventory_v1",
        "concept_id": concept,
        "short_name": config["collection"]["short_name"],
        "level": 3,
        "granules": records,
        "total_size_bytes": sum(record["size_bytes"] for record in records),
    }
    write_json(output, result)
    return result


def write_hycom_request(config: dict, output: Path) -> dict:
    source = config["hycom"]
    if "?" in source["source"]:
        raise ValueError("HYCOM source URLs with query strings are forbidden")
    request = {
        "source": source["source"],
        "variables": source["variables"],
        "start": source["start"],
        "end": source["end"],
        "bbox": source["bbox"],
        "depth": source["depth"],
        "coordinate_overrides": {},
        "dimension_bounds": {},
        "chunk_target_mib": source["chunk_target_mib"],
        "max_retries": source["max_retries"],
        "retry_delay_seconds": source["retry_delay_seconds"],
        "backoff": source["backoff"],
        "output": Path(config["paths"]["hycom_raw"]).name,
    }
    write_json(output, request)
    return request


def locate_hycom_fetcher() -> Path:
    raw = os.environ.get("HYCOM_FETCHER_SCRIPT", "")
    if not raw:
        raise RuntimeError("Set HYCOM_FETCHER_SCRIPT to the standard hycom_fetcher.py")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError("HYCOM_FETCHER_SCRIPT does not resolve to a file")
    return path


def run_hycom_inventory(config: dict, run_dir: Path) -> Path:
    script = locate_hycom_fetcher()
    output = run_dir / "hycom_inventory.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "inventory",
            "--source",
            config["hycom"]["source"],
            "--output",
            str(output),
        ],
        check=True,
    )
    return output


def run_hycom_fetch(config: dict, repository: Path, run_dir: Path) -> Path:
    script = locate_hycom_fetcher()
    request = run_dir / "hycom_request.json"
    write_hycom_request(config, request)
    plan = run_dir / "hycom_download_plan.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "estimate",
            "--request",
            str(request),
            "--run-dir",
            str(run_dir),
            "--output",
            str(plan),
        ],
        check=True,
    )
    output = (repository / config["paths"]["hycom_raw"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(script),
            "fetch",
            "--plan",
            str(plan),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
        ],
        check=True,
    )
    health = run_dir / "health_check.json"
    if not health.is_file():
        raise RuntimeError("HYCOM fetch completed without health_check.json")
    payload = json.loads(health.read_text(encoding="utf-8"))
    if str(payload.get("status", "")).lower() not in {"pass", "passed", "healthy", "ok"}:
        raise RuntimeError("HYCOM health check did not pass")
    return output


def download_podaac(config: dict, repository: Path, inventory: dict) -> list[Path]:
    if os.environ.get("PYCNOTIDE_EARTHDATA_ROTATED") != "1":
        raise RuntimeError(
            "Protected download blocked: rotate the exposed Earthdata credential, "
            "configure it outside the repository, then set PYCNOTIDE_EARTHDATA_ROTATED=1"
        )
    executable = shutil.which("podaac-data-downloader")
    if executable is None:
        raise RuntimeError("podaac-data-downloader is unavailable; install the Guam extra")
    target = (repository / config["paths"]["glider_raw"]).resolve()
    target.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for record in inventory["granules"]:
        subprocess.run(
            [
                executable,
                "-c",
                config["collection"]["short_name"],
                "-d",
                str(target),
                "-gr",
                record["filename"],
                "-e",
                ".nc",
            ],
            check=True,
        )
        path = target / record["filename"]
        if not path.is_file() or path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"Downloaded granule failed size closure: {record['filename']}")
        checksum = file_checksum(path, str(record["checksum_algorithm"]))
        if checksum.lower() != str(record["checksum"]).lower():
            raise RuntimeError(
                f"Downloaded granule failed CMR checksum closure: {record['filename']}"
            )
        outputs.append(path)
    return outputs
