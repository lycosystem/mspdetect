"""Mid-sagittal plane detection for one case."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import scipy as sp

from mspdetect import __version__
from mspdetect.config import DetectionConfig
from mspdetect.io import Case
from mspdetect.optimize import optimize
from mspdetect.plane import Plane
from mspdetect.preprocessing import SLICE_AXIS, CropBox, prepare_case

logger = logging.getLogger(__name__)

# Tolerances for considering the first start's plane equal to the best plane
AGREEMENT_ANGLE_DEG = 0.5
AGREEMENT_DISTANCE_MM = 0.5


@dataclass(frozen=True)
class GridResult:
    """Grid search, sorted from lowest to highest initial value."""

    planes: tuple[Plane, ...]
    values: np.ndarray
    deltas: np.ndarray


@dataclass(frozen=True)
class StartResult:
    """One optimisation run.

    ``initial_value`` is the quadratic loss of the grid plane the start began at,
    ``final_value`` the minimised Huber loss (with this start's ``delta``).
    ``trajectory`` holds the plane at every iterate (first = initial, last = final),
    ``huber`` and ``quadratic`` the corresponding objective values.
    """

    initial: Plane
    final: Plane
    delta: float
    initial_value: float
    final_value: float
    converged: bool
    message: str
    n_iter: int
    trajectory: tuple[Plane, ...]
    huber: np.ndarray
    quadratic: np.ndarray

    @property
    def final_quadratic(self) -> float:
        """Quadratic loss of the final plane (independent of δ)."""
        return float(self.quadratic[-1])


@dataclass(frozen=True)
class MSPResult:
    """Detected mid-sagittal plane of one case, with diagnostics.

    Attributes
    ----------
    plane
        The detected plane in world coordinates (see :class:`~mspdetect.plane.Plane`):
        the final plane of the start with the lowest final quadratic loss. With a
        single start, the result of that start.
    objective
        Final Huber objective of the chosen start.
    delta
        Huber threshold of the chosen start.
    best_start
        Index of the chosen start (0 = start from the best grid point).
    first_start_agrees
        Whether start 0 ended at the same plane as the chosen one, within
        ``AGREEMENT_ANGLE_DEG`` and ``AGREEMENT_DISTANCE_MM`` (distance measured at the
        bone's centre of mass). Always True for a single start.

    """

    case_id: str
    plane: Plane
    objective: float
    delta: float
    converged: bool
    best_start: int
    first_start_agrees: bool
    first_start_angle_deg: float
    first_start_distance_mm: float
    starts: tuple[StartResult, ...]
    grid: GridResult
    crop: CropBox
    config: DetectionConfig
    version: str
    runtime_s: float

    @property
    def first_start_plane(self) -> Plane:
        """Final plane of the start from the best grid point (``starts[0]``)."""
        return self.starts[0].final

    @property
    def crop_slices(self) -> tuple[int, int]:
        """Slices ``[start, stop)`` used to detect the plane (inferior to superior)."""
        return int(self.crop.start[SLICE_AXIS]), int(self.crop.stop[SLICE_AXIS])

    def select_plane(self, which: str) -> Plane:
        """``"first_start"``: :attr:`first_start_plane`; ``"best"``: :attr:`plane`."""
        if which == "first_start":
            return self.first_start_plane
        if which == "best":
            return self.plane
        raise ValueError(f"Unknown plane {which!r}; use 'first_start' or 'best'")


def detect_msp(case: Case, config: DetectionConfig | None = None) -> MSPResult:
    """Detect the mid-sagittal plane of a case.

    Parameters
    ----------
    case
        CT (and optional masks) as returned by :func:`~mspdetect.io.load_case`.
    config
        Detection settings; defaults to :class:`~mspdetect.config.DetectionConfig`.

    """
    config = config or DetectionConfig()
    start_time = time.perf_counter()
    prepared = prepare_case(case, config)
    grid, starts = optimize(prepared, config.optimizer)

    def to_plane(params) -> Plane:
        theta, phi, L = params
        return Plane.from_angles(theta, phi, L, prepared.R, prepared.t)

    start_results = tuple(
        StartResult(
            initial=to_plane(s.initial),
            final=to_plane(s.final),
            delta=s.delta,
            initial_value=float(grid.values[i]),
            final_value=s.final_value,
            converged=s.converged,
            message=s.message,
            n_iter=s.n_iter,
            trajectory=tuple(to_plane(p) for p in s.trajectory),
            huber=s.huber,
            quadratic=s.quadratic,
        )
        for i, s in enumerate(starts)
    )
    # Starts minimise Huber losses with different δ, so compare them by the quadratic loss
    best = int(np.argmin([s.final_quadratic for s in start_results]))
    plane = start_results[best].final

    com_frame = np.array(sp.ndimage.center_of_mass(prepared.bone)) * prepared.voxel_size
    com = prepared.R @ com_frame + prepared.t
    first = start_results[0].final
    angle = first.angle_to(plane)
    distance = abs(float(first.signed_distance(com) - plane.signed_distance(com)))

    result = MSPResult(
        case_id=case.case_id,
        plane=plane,
        objective=start_results[best].final_value,
        delta=start_results[best].delta,
        converged=start_results[best].converged,
        best_start=best,
        first_start_agrees=angle <= AGREEMENT_ANGLE_DEG and distance <= AGREEMENT_DISTANCE_MM,
        first_start_angle_deg=angle,
        first_start_distance_mm=distance,
        starts=start_results,
        grid=GridResult(
            planes=tuple(to_plane(p) for p in grid.params),
            values=grid.values,
            deltas=grid.deltas,
        ),
        crop=prepared.crop,
        config=config,
        version=__version__,
        runtime_s=time.perf_counter() - start_time,
    )
    theta, phi = plane.angles_deg
    logger.info(
        "%s: plane azimuth %.2f°, polar %.2f°, offset %.2f mm (%.1f s)",
        case.case_id,
        theta,
        phi,
        plane.offset,
        result.runtime_s,
    )
    return result
