"""Fitting the mid-sagittal plane by maximising the bilateral symmetry of bone.

For a candidate plane, every bone voxel is mirrored across the plane and the CT
intensity at the mirrored position is interpolated. The objective is the sum of a
loss of the differences between original and mirrored intensities.

1. Grid search: planes through the bone's centre of mass, for a grid of angles,
   scored with the quadratic loss. For each grid plane a robust Huber threshold δ is
   estimated from the residuals.
2. Optimisation: BFGS on the Huber loss, started from the ``n_starts`` best grid
   planes, each with the δ of its own grid plane. As the starts minimise different
   functions, their results are compared by the δ-independent quadratic loss.

Planes are parametrised as ``(theta, phi, L)``: normal ``n(theta, phi)`` (see
:func:`~mspdetect.plane.normal_from_angles`) and offset ``L`` (mm), in the crop frame
of :class:`~mspdetect.preprocessing.PreparedCase`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy as sp

from mspdetect.config import OptimizerConfig
from mspdetect.plane import normal_from_angles
from mspdetect.preprocessing import PreparedCase

logger = logging.getLogger(__name__)

# Robust scale: sigma = 1.4826 * MAD; Huber threshold with 95% efficiency: 1.345 * sigma
MAD_TO_SIGMA = 1.4826
HUBER_EFFICIENCY = 1.345


def quadratic_loss(diff: np.ndarray) -> np.ndarray:
    """``0.5 * diff**2``."""
    return 0.5 * diff**2


def huber_loss(diff: np.ndarray, delta: float) -> np.ndarray:
    """Quadratic for ``|diff| <= delta``, linear beyond."""
    a = np.abs(diff)
    return np.where(a <= delta, 0.5 * diff**2, delta * (a - 0.5 * delta))


def huber_delta(diff: np.ndarray) -> float:
    """Huber threshold from the median absolute deviation of the (centred) residuals."""
    mad = np.median(np.abs(diff - diff.mean()))
    return float(HUBER_EFFICIENCY * MAD_TO_SIGMA * mad)


class SymmetryObjective:
    """Mirror-symmetry objective of the bone voxels of a prepared case."""

    def __init__(self, prepared: PreparedCase):
        indices = np.nonzero(prepared.bone)
        self.points = np.column_stack(indices) * prepared.voxel_size
        self.values = prepared.bone[indices].astype(float)
        self.interpolator = prepared.interpolator

    def residuals(self, params) -> np.ndarray:
        """Original minus mirrored intensity for each bone voxel."""
        theta, phi, L = params
        n = normal_from_angles(theta, phi)
        d = self.points @ n - L
        mirrored = self.points - 2 * d[:, None] * n
        return self.values - self.interpolator(mirrored)

    def quadratic(self, params) -> float:
        """Sum of the quadratic loss."""
        return float(quadratic_loss(self.residuals(params)).sum())

    def huber(self, params, delta: float) -> float:
        """Sum of the Huber loss."""
        return float(huber_loss(self.residuals(params), delta).sum())


@dataclass(frozen=True)
class GridSearch:
    """All grid planes with their quadratic loss and Huber δ, sorted best first."""

    params: np.ndarray  # (N, 3) theta, phi, L
    values: np.ndarray  # (N,)
    deltas: np.ndarray  # (N,)


@dataclass(frozen=True)
class Start:
    """One BFGS optimisation.

    ``trajectory``, ``huber`` and ``quadratic`` hold one entry per iterate, starting
    with the initial parameters and ending with the final ones.
    """

    initial: np.ndarray
    final: np.ndarray
    delta: float
    final_value: float
    converged: bool
    message: str
    n_iter: int
    trajectory: np.ndarray  # (k, 3)
    huber: np.ndarray  # (k,)
    quadratic: np.ndarray  # (k,)


def grid_search(
    objective: SymmetryObjective, prepared: PreparedCase, config: OptimizerConfig
) -> GridSearch:
    """Score planes through the bone's centre of mass on a grid of angles."""
    com = np.array(sp.ndimage.center_of_mass(prepared.bone)) * prepared.voxel_size

    def angles(range_deg):
        centre, half = np.deg2rad(range_deg)
        return np.linspace(centre - half, centre + half, config.grid_steps)

    theta, phi = np.meshgrid(angles(config.azimuth_deg), angles(config.polar_deg), indexing="ij")
    theta, phi = theta.ravel(), phi.ravel()
    L = normal_from_angles(theta, phi) @ com
    params = np.column_stack([theta, phi, L])

    values, deltas = np.empty(len(params)), np.empty(len(params))
    for i, p in enumerate(params):
        diff = objective.residuals(p)
        values[i] = quadratic_loss(diff).sum()
        deltas[i] = huber_delta(diff)

    order = np.argsort(values, kind="stable")
    return GridSearch(params[order], values[order], deltas[order])


def run_bfgs(objective: SymmetryObjective, initial, delta: float) -> Start:
    """Minimise the Huber loss with BFGS, recording every iterate."""
    trajectory = [np.array(initial, dtype=float)]
    huber = [objective.huber(initial, delta)]
    quadratic = [objective.quadratic(initial)]

    def record(xk):
        trajectory.append(xk.copy())
        huber.append(objective.huber(xk, delta))
        quadratic.append(objective.quadratic(xk))

    res = sp.optimize.minimize(
        objective.huber, x0=initial, args=(delta,), method="BFGS", callback=record
    )
    if not np.array_equal(trajectory[-1], res.x):
        record(res.x)
    return Start(
        initial=np.array(initial, dtype=float),
        final=res.x.copy(),
        delta=float(delta),
        final_value=float(res.fun),
        converged=bool(res.success),
        message=str(res.message),
        n_iter=int(res.nit),
        trajectory=np.array(trajectory),
        huber=np.array(huber),
        quadratic=np.array(quadratic),
    )


def optimize(prepared: PreparedCase, config: OptimizerConfig) -> tuple[GridSearch, list[Start]]:
    """Grid search followed by ``config.n_starts`` BFGS runs.

    Start ``i`` begins at the ``i``-th best grid plane and uses that plane's δ, so
    ``starts[0]`` is the start from the grid plane with the lowest quadratic loss.
    """
    objective = SymmetryObjective(prepared)
    grid = grid_search(objective, prepared, config)
    logger.debug("Best grid value %.4g, delta %.4g", grid.values[0], grid.deltas[0])
    starts = [run_bfgs(objective, grid.params[i], grid.deltas[i]) for i in range(config.n_starts)]
    return grid, starts
