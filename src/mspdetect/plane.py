"""Plane geometry.

A :class:`Plane` is always stored in **world coordinates** (scanner RAS+, mm), as
defined by the NIfTI affine: +x points to the patient's right, +y anterior, +z
superior. The plane is the set of points ``w`` with ``normal · w = offset``.

The normal is a unit vector oriented towards the patient's right (``normal[0] >= 0``),
so the signed distance ``normal · w - offset`` is positive on the patient's right and
negative on the patient's left, independent of how the image was stored on disk.

Internally the optimizer works in a *frame*: a Cartesian mm coordinate system that is
related to world coordinates by ``w = R @ p + t``. :meth:`Plane.from_frame` and
:meth:`Plane.to_frame` convert between the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normal_from_angles(theta, phi) -> np.ndarray:
    """Unit normal(s) from azimuthal angle ``theta`` and polar angle ``phi`` (radians).

    ``n = (sin(phi) cos(theta), sin(phi) sin(theta), cos(phi))``; array inputs broadcast
    and the components are stacked along the last axis.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        axis=-1,
    )


def angles_from_normal(normal) -> tuple[float, float]:
    """Azimuthal and polar angle (radians) of a (not necessarily unit) vector."""
    x, y, z = normal
    return float(np.arctan2(y, x)), float(np.arctan2(np.hypot(x, y), z))


def array_frame(affine: np.ndarray, start_index=(0, 0, 0)) -> tuple[np.ndarray, np.ndarray]:
    """Frame of an image array in mm, with its origin at voxel ``start_index``.

    Frame coordinates are ``p = (index - start_index) * voxel_size``, i.e. the array
    axes scaled to mm. Returns ``(R, t)`` such that ``world = R @ p + t``.
    """
    affine = np.asarray(affine, dtype=float)
    linear = affine[:3, :3]
    voxel_size = np.linalg.norm(linear, axis=0)
    R = linear / voxel_size
    t = linear @ np.asarray(start_index, dtype=float) + affine[:3, 3]
    return R, t


@dataclass(frozen=True, eq=False)
class Plane:
    """A plane ``{w : normal · w = offset}`` in world coordinates (RAS+, mm).

    The normal is normalised and oriented towards the patient's right on creation.
    """

    normal: np.ndarray
    offset: float

    def __post_init__(self):
        normal = np.asarray(self.normal, dtype=float).reshape(3)
        length = np.linalg.norm(normal)
        if not np.isfinite(length) or length == 0:
            raise ValueError(f"Invalid plane normal: {self.normal}")
        offset = float(self.offset)
        if abs(length - 1) > 1e-12:  # keep unit normals exact, e.g. when reloading
            normal, offset = normal / length, offset / length
        if normal[0] < 0:
            normal, offset = -normal, -offset
        normal.flags.writeable = False
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "offset", offset)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Plane):
            return NotImplemented
        return bool(np.array_equal(self.normal, other.normal) and self.offset == other.offset)

    def __hash__(self) -> int:
        return hash((*self.normal, self.offset))

    def signed_distance(self, points) -> np.ndarray:
        """Signed distance (mm) of world points of shape ``(..., 3)``.

        Positive on the patient's right, negative on the patient's left.
        """
        return np.asarray(points, dtype=float) @ self.normal - self.offset

    def signed_distance_voxels(self, indices, affine) -> np.ndarray:
        """Signed distance (mm) of voxel ``indices`` (shape ``(N, 3)``) of an image."""
        affine = np.asarray(affine, dtype=float)
        m = affine[:3, :3].T @ self.normal
        c = affine[:3, 3] @ self.normal - self.offset
        return np.asarray(indices, dtype=float) @ m + c

    @property
    def angles_deg(self) -> tuple[float, float]:
        """Azimuthal and polar angle (degrees) of the normal in world coordinates.

        A plane exactly aligned with the scanner's sagittal plane has ``(0, 90)``.
        """
        theta, phi = angles_from_normal(self.normal)
        return float(np.rad2deg(theta)), float(np.rad2deg(phi))

    @classmethod
    def from_frame(cls, normal, offset, R, t) -> Plane:
        """Plane ``normal · p = offset`` given in a frame with ``world = R @ p + t``."""
        m = np.linalg.solve(np.asarray(R, dtype=float).T, np.asarray(normal, dtype=float))
        return cls(m, float(offset) + m @ np.asarray(t, dtype=float))

    def to_frame(self, R, t) -> tuple[np.ndarray, float]:
        """Unit normal and offset of this plane in a frame with ``world = R @ p + t``."""
        m = np.asarray(R, dtype=float).T @ self.normal
        offset = self.offset - self.normal @ np.asarray(t, dtype=float)
        length = np.linalg.norm(m)
        return m / length, offset / length

    @classmethod
    def from_angles(cls, theta, phi, offset, R=None, t=None) -> Plane:
        """Plane from spherical angles (radians) and offset, optionally in a frame."""
        normal = normal_from_angles(theta, phi)
        if R is None:
            return cls(normal, offset)
        return cls.from_frame(normal, offset, R, t)

    def angle_to(self, other: Plane) -> float:
        """Angle (degrees) between the normals of two planes."""
        cos = np.clip(abs(self.normal @ other.normal), -1.0, 1.0)
        return float(np.rad2deg(np.arccos(cos)))

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        theta, phi = self.angles_deg
        return {
            "normal": [float(v) for v in self.normal],
            "offset_mm": self.offset,
            "azimuth_deg": theta,
            "polar_deg": phi,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Plane:
        """Inverse of :meth:`to_dict`."""
        return cls(data["normal"], data["offset_mm"])

    def as_array(self) -> np.ndarray:
        """``[nx, ny, nz, offset]``."""
        return np.append(self.normal, self.offset)
