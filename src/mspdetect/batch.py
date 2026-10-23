"""Processing many cases, as used by the command line interface.

Every case is processed independently; a failing case is recorded in a failures
table and does not stop the run. Results are written to ``<output>/<case>/``.
"""

from __future__ import annotations

import csv
import logging
import multiprocessing
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from mspdetect import __version__
from mspdetect.config import BatchConfig
from mspdetect.detect import detect_msp
from mspdetect.io import load_case
from mspdetect.metrics import compute_metrics
from mspdetect.results import (
    MANUAL_FILE,
    SUMMARY_FILE,
    load_manual_plane,
    load_result,
    read_summary,
    save_result,
)

logger = logging.getLogger(__name__)

RESULTS_TABLE = "msp_results.csv"
FAILURES_TABLE = "failures.csv"
METRICS_TABLE = "metrics.csv"
METRICS_FAILURES_TABLE = "metrics_failures.csv"

# Structures used by the detection itself (besides the crop reference)
DETECTION_STRUCTURES = ("Body", "Mandible")


def list_cases(config: BatchConfig) -> list[str]:
    """Case names (paths relative to ``data.root``) to process."""
    root, patients = config.data.root, config.data.patients
    if not root.is_dir():
        raise FileNotFoundError(f"data.root is not a directory: {root}")
    if patients is None:
        return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if isinstance(patients, Path):
        lines = patients.read_text().splitlines()
        return [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return list(patients)


def _structure_files(config: BatchConfig, names: Iterable[str]) -> dict:
    return {n: config.data.structures[n] for n in names if n in config.data.structures}


def _detection_structures(config: BatchConfig) -> dict:
    names = [*DETECTION_STRUCTURES]
    if config.detection.crop.reference is not None:
        names.append(config.detection.crop.reference)
        if config.detection.crop.reference not in config.data.structures:
            raise ValueError(
                f"Crop reference '{config.detection.crop.reference}' is not in data.structures",
            )
    return _structure_files(config, names)


def _detect_one(case_name: str, config: BatchConfig) -> dict:
    """Detect and save the plane of one case; returns a status row."""
    out_dir = config.output / case_name
    fingerprint = config.detection.fingerprint()
    if (out_dir / SUMMARY_FILE).exists() and not config.overwrite:
        existing = read_summary(out_dir)
        if existing.get("config_fingerprint") != fingerprint:
            raise RuntimeError(
                "existing result was computed with different detection settings; "
                "use another output directory or set overwrite: true",
            )
        if existing.get("version") != __version__:
            logger.warning(
                "%s: existing result from mspdetect %s (now %s) is kept",
                case_name,
                existing.get("version"),
                __version__,
            )
        return {"case_id": case_name, "status": "existing"}

    case = load_case(
        config.data.root / case_name,
        image=config.data.image,
        structures=_detection_structures(config),
        case_id=case_name,
    )
    save_result(detect_msp(case, config.detection), out_dir)
    return {"case_id": case_name, "status": "done"}


def _metrics_starts(result, which: str) -> list[int]:
    """Indices of the starts whose planes are evaluated for ``metrics.plane``."""
    if which == "first_start":
        return [0]
    if which == "best":
        return [result.best_start]
    return list(range(len(result.starts)))


def _metrics_one(case_name: str, config: BatchConfig) -> list[dict]:
    """Compute the metrics of one case from its saved plane(s); one row per plane.

    A manual correction (``manual.json``) replaces the detected plane(s).
    """
    result = load_result(config.output / case_name)
    manual = load_manual_plane(config.output / case_name)
    names = [*config.metrics.structures]
    if config.metrics.target is not None:
        names.append(config.metrics.target)
    case = load_case(
        config.data.root / case_name,
        image=config.data.image,
        structures=_structure_files(config, names),
        case_id=case_name,
    )
    slices = result.crop_slices if config.metrics.volume_region == "slab" else None
    if manual is not None:
        planes = [("manual", manual.plane)]
    else:
        planes = [
            (i, result.starts[i].final) for i in _metrics_starts(result, config.metrics.plane)
        ]
    rows = []
    for start, plane in planes:
        row = compute_metrics(
            case,
            plane,
            config.metrics.structures,
            config.metrics.target,
            slices,
        )
        rows.append(
            {
                "case_id": row.pop("case_id"),
                "plane_source": "detected" if manual is None else "manual",
                "plane": config.metrics.plane,
                "start": start,
                "is_first_start": start == 0,
                "is_best": start == result.best_start,
                "volume_region": config.metrics.volume_region,
                **row,
            },
        )
    return rows


def _safe(
    func: Callable, case_name: str, config: BatchConfig
) -> tuple[str, dict | list[dict] | None, str | None]:
    try:
        return case_name, func(case_name, config), None
    except Exception as e:  # noqa: BLE001 - record any failure and continue with other cases
        logger.exception("%s failed", case_name)
        return case_name, None, f"{type(e).__name__}: {e}"


def _run_all(func: Callable, cases: list[str], config: BatchConfig, description: str):
    """``(case, row, error)`` for every case, in the order of ``cases``.

    Cases run in parallel if ``n_jobs > 1``; a progress bar counts finished cases.
    """
    outcomes = {}
    with logging_redirect_tqdm(), tqdm(total=len(cases), desc=description, unit="case") as bar:
        if config.n_jobs == 1:
            finished = (_safe(func, c, config) for c in cases)
            pool = None
        else:
            # "spawn": forking a process with running threads (e.g. tqdm's) can deadlock
            pool = ProcessPoolExecutor(
                max_workers=config.n_jobs, mp_context=multiprocessing.get_context("spawn")
            )
            futures = [pool.submit(_safe, func, c, config) for c in cases]
            finished = (f.result() for f in as_completed(futures))
        try:
            for case_name, row, error in finished:
                outcomes[case_name] = (case_name, row, error)
                bar.update()
        finally:
            if pool is not None:
                pool.shutdown(cancel_futures=True)
    return [outcomes[c] for c in cases]


def write_table(path: Path, rows: list[dict], columns: Iterable[str] = ()) -> None:
    """Write rows (dicts) as CSV; columns are ``columns`` followed by any other keys."""
    columns = list(dict.fromkeys([*columns, *(k for row in rows for k in row)]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(case_name: str, summary: dict, config_output: Path) -> dict:
    plane = summary["plane"]
    return {
        "case_id": case_name,
        "normal_x": plane["normal"][0],
        "normal_y": plane["normal"][1],
        "normal_z": plane["normal"][2],
        "offset_mm": plane["offset_mm"],
        "azimuth_deg": plane["azimuth_deg"],
        "polar_deg": plane["polar_deg"],
        "objective": summary["objective"],
        "quadratic": summary["quadratic"],
        "converged": summary["converged"],
        "n_starts": len(summary["starts"]),
        "best_start": summary["best_start"],
        "first_start_agrees": summary["first_start_agrees"] if len(summary["starts"]) > 1 else "",
        "first_start_angle_deg": summary["first_start_angle_deg"]
        if len(summary["starts"]) > 1
        else "",
        "first_start_distance_mm": summary["first_start_distance_mm"]
        if len(summary["starts"]) > 1
        else "",
        "runtime_s": summary["runtime_s"],
        "version": summary["version"],
        "manual_correction": (config_output / case_name / MANUAL_FILE).exists(),
    }


def run_detection(config: BatchConfig) -> tuple[int, int]:
    """Detect the plane of all cases; returns ``(n_succeeded, n_failed)``.

    Writes ``msp_results.csv`` (all cases with a result) and ``failures.csv``.
    """
    cases = list_cases(config)
    _detection_structures(config)  # fail early on configuration errors
    config.output.mkdir(parents=True, exist_ok=True)
    logger.info("Detecting the MSP of %d case(s) with %d job(s)", len(cases), config.n_jobs)

    rows, failures = [], []
    for case_name, _, error in _run_all(_detect_one, cases, config, "Detecting MSP"):
        if error:
            failures.append({"case_id": case_name, "error": error})
        else:
            rows.append(
                _summary_row(case_name, read_summary(config.output / case_name), config.output)
            )

    write_table(config.output / RESULTS_TABLE, rows)
    write_table(config.output / FAILURES_TABLE, failures, columns=("case_id", "error"))
    logger.info(
        "Done: %d succeeded, %d failed (see %s)",
        len(rows),
        len(failures),
        FAILURES_TABLE,
    )
    return len(rows), len(failures)


def run_metrics(config: BatchConfig) -> tuple[int, int]:
    """Compute metrics for all cases with a detected plane.

    Planes are read from ``output``; ``metrics.csv`` and ``metrics_failures.csv`` are
    written to ``metrics.output`` (default: ``output``).
    """
    cases = [c for c in list_cases(config) if (config.output / c / SUMMARY_FILE).exists()]
    out_dir = config.metrics_output
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Computing metrics of %d case(s), writing to %s", len(cases), out_dir)

    rows, failures = [], []
    for case_name, case_rows, error in _run_all(_metrics_one, cases, config, "Computing metrics"):
        if error:
            failures.append({"case_id": case_name, "error": error})
        else:
            rows.extend(case_rows)

    write_table(out_dir / METRICS_TABLE, rows)
    write_table(out_dir / METRICS_FAILURES_TABLE, failures, columns=("case_id", "error"))
    n_done = len(cases) - len(failures)
    logger.info("Done: %d succeeded, %d failed", n_done, len(failures))
    return n_done, len(failures)
