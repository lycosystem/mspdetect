"""Saving and loading detection results.

A result directory contains

- ``msp.json``: the plane, settings, package version and a summary of every start;
- ``details.npz``: the grid search and the per-iteration trajectories of every start;
- ``manual.json`` (optional): a manually corrected plane with a note, date and user.
  It never changes the two files above; deleting it restores the detected plane.

Planes are stored as ``[nx, ny, nz, offset_mm]`` in world coordinates (RAS+).
``msp.json`` is written last, so its presence marks a complete result.
"""

from __future__ import annotations

import datetime
import getpass
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mspdetect.config import DetectionConfig
from mspdetect.detect import GridResult, MSPResult, StartResult
from mspdetect.plane import Plane
from mspdetect.preprocessing import CropBox

SUMMARY_FILE = "msp.json"
DETAILS_FILE = "details.npz"
MANUAL_FILE = "manual.json"


def _planes_array(planes) -> np.ndarray:
    return np.array([p.as_array() for p in planes])


def _planes_from_array(array) -> tuple[Plane, ...]:
    return tuple(Plane(row[:3], row[3]) for row in np.asarray(array))


def result_summary(result: MSPResult) -> dict:
    """JSON-serialisable summary of a result (content of ``msp.json``)."""
    return {
        "case_id": result.case_id,
        "version": result.version,
        "plane": result.plane.to_dict(),
        "objective": result.objective,
        "quadratic": result.starts[result.best_start].final_quadratic,
        "delta": result.delta,
        "converged": result.converged,
        "best_start": result.best_start,
        "first_start_agrees": result.first_start_agrees,
        "first_start_angle_deg": result.first_start_angle_deg,
        "first_start_distance_mm": result.first_start_distance_mm,
        "crop": result.crop.to_dict(),
        "runtime_s": result.runtime_s,
        "config_fingerprint": result.config.fingerprint(),
        "config": result.config.model_dump(mode="json"),
        "starts": [
            {
                "initial": s.initial.to_dict(),
                "final": s.final.to_dict(),
                "delta": s.delta,
                "initial_value": s.initial_value,
                "final_value": s.final_value,
                "final_quadratic": s.final_quadratic,
                "converged": s.converged,
                "message": s.message,
                "n_iter": s.n_iter,
            }
            for s in result.starts
        ],
    }


def save_result(result: MSPResult, directory: str | Path) -> Path:
    """Write ``msp.json`` and ``details.npz`` to ``directory`` (created if needed)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    arrays = {
        "grid_planes": _planes_array(result.grid.planes),
        "grid_values": result.grid.values,
        "grid_deltas": result.grid.deltas,
    }
    for i, s in enumerate(result.starts):
        arrays[f"start{i:02d}_trajectory"] = _planes_array(s.trajectory)
        arrays[f"start{i:02d}_huber"] = s.huber
        arrays[f"start{i:02d}_quadratic"] = s.quadratic
    tmp = directory / f".{DETAILS_FILE}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, directory / DETAILS_FILE)

    tmp = directory / f".{SUMMARY_FILE}.tmp"
    tmp.write_text(json.dumps(result_summary(result), indent=2))
    os.replace(tmp, directory / SUMMARY_FILE)
    return directory


def read_summary(directory: str | Path) -> dict:
    """Content of ``msp.json`` in ``directory``."""
    return json.loads((Path(directory) / SUMMARY_FILE).read_text())


def load_result(directory: str | Path) -> MSPResult:
    """Load a result written by :func:`save_result`."""
    directory = Path(directory)
    summary = read_summary(directory)
    with np.load(directory / DETAILS_FILE, allow_pickle=False) as details:
        arrays = {k: details[k] for k in details.files}

    starts = tuple(
        StartResult(
            initial=Plane.from_dict(s["initial"]),
            final=Plane.from_dict(s["final"]),
            delta=s["delta"],
            initial_value=s["initial_value"],
            final_value=s["final_value"],
            converged=s["converged"],
            message=s["message"],
            n_iter=s["n_iter"],
            trajectory=_planes_from_array(arrays[f"start{i:02d}_trajectory"]),
            huber=arrays[f"start{i:02d}_huber"],
            quadratic=arrays[f"start{i:02d}_quadratic"],
        )
        for i, s in enumerate(summary["starts"])
    )
    return MSPResult(
        case_id=summary["case_id"],
        plane=Plane.from_dict(summary["plane"]),
        objective=summary["objective"],
        delta=summary["delta"],
        converged=summary["converged"],
        best_start=summary["best_start"],
        first_start_agrees=summary["first_start_agrees"],
        first_start_angle_deg=summary["first_start_angle_deg"],
        first_start_distance_mm=summary["first_start_distance_mm"],
        starts=starts,
        grid=GridResult(
            planes=_planes_from_array(arrays["grid_planes"]),
            values=arrays["grid_values"],
            deltas=arrays["grid_deltas"],
        ),
        crop=CropBox.from_dict(summary["crop"]),
        config=DetectionConfig.model_validate(summary["config"]),
        version=summary["version"],
        runtime_s=summary["runtime_s"],
    )


@dataclass(frozen=True)
class ManualPlane:
    """A manually corrected plane and why, when and by whom it was corrected."""

    plane: Plane
    detected_plane: Plane
    note: str
    created: str
    user: str
    version: str
    adjustment: dict


def save_manual_plane(
    directory: str | Path,
    plane: Plane,
    note: str,
    detected_plane: Plane,
    adjustment: dict | None = None,
) -> Path:
    """Store a manual correction of the plane in ``directory`` as ``manual.json``.

    Parameters
    ----------
    directory
        Result directory; the detected result (``msp.json``) must exist and is not
        modified.
    plane
        The corrected plane.
    note
        Why the plane was corrected (required).
    detected_plane
        The detected plane that was corrected, kept for reference.
    adjustment
        How the plane was changed, e.g. rotations and shift from
        :func:`mspdetect.viz.adjust_plane`.

    An existing correction is replaced.
    """
    from mspdetect import __version__

    directory = Path(directory)
    if not note or not note.strip():
        raise ValueError("A note explaining the manual correction is required")
    if not (directory / SUMMARY_FILE).exists():
        raise ValueError(f"No detected result ({SUMMARY_FILE}) in {directory}")
    data = {
        "plane": plane.to_dict(),
        "detected_plane": detected_plane.to_dict(),
        "note": note.strip(),
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "user": getpass.getuser(),
        "version": __version__,
        "adjustment": adjustment or {},
    }
    path = directory / MANUAL_FILE
    tmp = directory / f".{MANUAL_FILE}.tmp"
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)
    return path


def load_manual_plane(directory: str | Path) -> ManualPlane | None:
    """The manual correction in ``directory``, or None if there is none."""
    path = Path(directory) / MANUAL_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return ManualPlane(
        plane=Plane.from_dict(data["plane"]),
        detected_plane=Plane.from_dict(data["detected_plane"]),
        note=data["note"],
        created=data["created"],
        user=data["user"],
        version=data["version"],
        adjustment=data.get("adjustment", {}),
    )
