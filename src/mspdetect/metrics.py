"""Metrics of structures relative to the mid-sagittal plane.

Distances are signed distances of voxel centres to the plane: positive on the
patient's right, negative on the patient's left (see :class:`~mspdetect.plane.Plane`).
Volumes are in cc (= cm³ = ml), distances in mm.

Left/right volumes are computed either for the whole structure or only within a range
of slices, typically the slices used to detect the plane
(:attr:`~mspdetect.detect.MSPResult.crop_slices`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist

from mspdetect.io import Case
from mspdetect.plane import Plane
from mspdetect.preprocessing import SLICE_AXIS


def _voxels(case: Case, structure: str, slices: tuple[int, int] | None) -> np.ndarray:
    indices = np.argwhere(case.masks[structure])
    if slices is not None:
        z = indices[:, SLICE_AXIS]
        indices = indices[(z >= slices[0]) & (z < slices[1])]
    return indices


def signed_distances(
    case: Case,
    plane: Plane,
    structure: str,
    slices: tuple[int, int] | None = None,
) -> np.ndarray:
    """Signed distance (mm) of every voxel of ``structure`` (within ``slices``) to ``plane``."""
    return plane.signed_distance_voxels(_voxels(case, structure, slices), case.affine)


def side_volumes(
    case: Case,
    plane: Plane,
    structure: str,
    slices: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """Volume (cc) of ``structure`` on the patient's left and right of ``plane``.

    With ``slices = (start, stop)``, only the part of the structure within these
    slices is counted. Voxels whose centre lies exactly on the plane are not counted.
    """
    d = signed_distances(case, plane, structure, slices)
    cc = case.voxel_volume / 1000  # mm³ -> cc
    return float(np.count_nonzero(d < 0) * cc), float(np.count_nonzero(d > 0) * cc)


def max_diameter(case: Case, structure: str) -> float:
    """Largest distance (mm) between two voxel centres of ``structure``."""
    points = case.voxel_to_world(np.argwhere(case.masks[structure]))
    if len(points) < 2:
        return 0.0
    try:
        points = points[ConvexHull(points).vertices]
    except QhullError:
        pass  # degenerate (e.g. flat) structure: use all points
    return float(pdist(points).max())


def target_metrics(case: Case, plane: Plane, structure: str = "GTVp") -> dict:
    """Laterality and extent of a target structure relative to the plane.

    Returns
    -------
    dict
        - ``side``: ``"left"``/``"right"``: side containing the larger volume
          (``"central"`` if equal).
        - ``crosses_midline``: whether the structure has voxels on both sides.
        - ``distance_msp_mm``: if crossing, how far the structure extends into the
          side with the smaller volume (positive); otherwise minus the gap between
          the plane and the closest voxel (negative).
        - ``lateral_extent_mm``: extent of the structure perpendicular to the plane.
        - ``left_max_mm`` / ``right_max_mm``: largest distance from the plane on each
          side (0 if no voxels on that side).
        - ``max_diameter_mm``: largest distance between two voxels.

    """
    d = signed_distances(case, plane, structure)
    if d.size == 0:
        raise ValueError(f"{case.case_id}: structure '{structure}' is empty")
    left, right = -d[d < 0], d[d > 0]

    if left.size > right.size:
        side = "left"
    elif right.size > left.size:
        side = "right"
    else:
        side = "central"

    crosses = left.size > 0 and right.size > 0
    if crosses:
        distance = left.max() if side in ("right", "central") else right.max()
        extent = left.max() + right.max()
    else:
        on_side = left if left.size else right
        distance = -on_side.min() if on_side.size else 0.0
        extent = on_side.max() - on_side.min() if on_side.size else 0.0

    return {
        "side": side,
        "crosses_midline": bool(crosses),
        "distance_msp_mm": float(distance),
        "lateral_extent_mm": float(extent),
        "left_max_mm": float(left.max()) if left.size else 0.0,
        "right_max_mm": float(right.max()) if right.size else 0.0,
        "max_diameter_mm": max_diameter(case, structure),
    }


def compute_metrics(
    case: Case,
    plane: Plane,
    structures: Sequence[str] = (),
    target: str | None = "GTVp",
    slices: tuple[int, int] | None = None,
) -> dict:
    """All metrics of a case as a flat dictionary (one row of a results table).

    Parameters
    ----------
    case
        Case with the masks of ``structures`` and ``target``.
    plane
        Detected plane, e.g. ``detect_msp(case).plane``.
    structures
        Structures whose left/right volumes are computed. Missing structures give NaN.
    target
        Structure for :func:`target_metrics` (columns prefixed with its name), or None.
        Target metrics always use the whole target.
    slices
        ``(start, stop)``: count volumes only within these slices (full left-right and
        anterior-posterior extent), e.g. ``result.crop_slices`` for the slices used to
        detect the plane. None: whole structures.

    """
    if target is not None and target not in case.masks:
        raise ValueError(f"{case.case_id}: target structure '{target}' not loaded")
    row = {"case_id": case.case_id}
    if slices is not None:
        row["slab_start"], row["slab_stop"] = int(slices[0]), int(slices[1])

    for name in structures:
        if name in case.masks:
            row[f"{name}_left_cc"], row[f"{name}_right_cc"] = side_volumes(
                case, plane, name, slices
            )
        else:
            row[f"{name}_left_cc"] = row[f"{name}_right_cc"] = float("nan")
    if target is not None:
        row.update({f"{target}_{k}": v for k, v in target_metrics(case, plane, target).items()})
    return row
