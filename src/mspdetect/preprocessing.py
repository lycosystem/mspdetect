"""Cropping and bone extraction before the plane is fitted.

All functions work on a :class:`~mspdetect.io.Case` in canonical orientation: axis 0
is left→right (lateral), axis 2 is inferior→superior (slices).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import scipy as sp

from mspdetect.config import CropConfig, DetectionConfig
from mspdetect.io import Case
from mspdetect.plane import array_frame

logger = logging.getLogger(__name__)

LATERAL_AXIS = 0
SLICE_AXIS = 2
# Slices of the mandible included when the reference ends just below it
MANDIBLE_SLICES = 5


@dataclass(frozen=True)
class CropBox:
    """Voxel box ``[start, stop)`` along each array axis."""

    start: tuple[int, int, int]
    stop: tuple[int, int, int]

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        """Index expression selecting the box from an array."""
        return tuple(slice(a, b) for a, b in zip(self.start, self.stop, strict=True))

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the cropped array."""
        return tuple(b - a for a, b in zip(self.start, self.stop, strict=True))

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        return {"start": list(self.start), "stop": list(self.stop)}

    @classmethod
    def from_dict(cls, data: dict) -> CropBox:
        """Inverse of :meth:`to_dict`."""
        return cls(tuple(data["start"]), tuple(data["stop"]))


@dataclass(frozen=True)
class PreparedCase:
    """Input of the optimizer.

    Coordinates are in the *crop frame*: array axes in mm with the origin at the first
    voxel of ``crop``. ``R`` and ``t`` map them to world coordinates
    (``world = R @ p + t``).
    """

    bone: np.ndarray
    interpolator: sp.interpolate.RegularGridInterpolator
    voxel_size: np.ndarray
    crop: CropBox
    R: np.ndarray
    t: np.ndarray


def _extent(mask: np.ndarray, axis: int) -> tuple[int, int] | None:
    """First and last+1 index along ``axis`` where ``mask`` is non-zero."""
    other = tuple(a for a in range(mask.ndim) if a != axis)
    idx = np.flatnonzero(mask.any(axis=other))
    return (int(idx[0]), int(idx[-1]) + 1) if idx.size else None


def slice_bounds(case: Case, crop: CropConfig) -> tuple[int, int]:
    """Slice range ``[start, stop)`` used to fit the plane."""
    n_slices = case.image.shape[SLICE_AXIS]
    if crop.slice_range is not None:
        start, stop = crop.slice_range
    elif crop.reference is None:
        start, stop = 0, n_slices
    else:
        if crop.reference not in case.masks:
            raise ValueError(
                f"{case.case_id}: crop reference '{crop.reference}' not loaded. Provide "
                "it, or set crop.reference to null to use the whole scan.",
            )
        extent = _extent(case.masks[crop.reference], SLICE_AXIS)
        if extent is None:
            raise ValueError(f"{case.case_id}: crop reference '{crop.reference}' is empty")
        start, stop = extent

        if crop.include_mandible and "Mandible" in case.masks:
            mandible = _extent(case.masks["Mandible"], SLICE_AXIS)
            if mandible is not None and stop - 1 <= mandible[0] + MANDIBLE_SLICES:
                start = min(start, mandible[0])
                stop = max(stop, mandible[0] + MANDIBLE_SLICES + 1)

        if crop.mode == "margins":
            below, above = (
                (crop.margins, crop.margins) if isinstance(crop.margins, int) else crop.margins
            )
            start, stop = start - below, stop + above
        elif stop - 1 - start <= 2 * crop.fixed_size:
            centre = (start + stop - 1) // 2
            start, stop = centre - crop.fixed_size, centre + crop.fixed_size + 1

    start, stop = max(0, start), min(n_slices, stop)
    if start >= stop:
        raise ValueError(f"{case.case_id}: empty slice range [{start}, {stop})")
    return start, stop


def lateral_bounds(case: Case, crop: CropConfig) -> tuple[int, int]:
    """Left-right range ``[start, stop)`` used to fit the plane (shoulder removal)."""
    n = case.image.shape[LATERAL_AXIS]
    if crop.lateral_extent_mm is None:
        return 0, n
    extent = _extent(case.masks["Body"], LATERAL_AXIS) if "Body" in case.masks else None
    centre = (extent[0] + extent[1] - 1) // 2 if extent else n // 2
    half = int(crop.lateral_extent_mm / case.voxel_size[LATERAL_AXIS]) // 2
    return max(0, centre - half), min(n, centre + half)


def crop_box(case: Case, crop: CropConfig) -> CropBox:
    """Box of the case used to fit the plane."""
    x0, x1 = lateral_bounds(case, crop)
    z0, z1 = slice_bounds(case, crop)
    return CropBox((x0, 0, z0), (x1, case.image.shape[1], z1))


def bone_image(image: np.ndarray, hu_range, body: np.ndarray | None = None) -> np.ndarray:
    """Keep voxels with HU in ``hu_range`` (inside ``body``, if given); zero elsewhere."""
    if body is not None:
        image = np.where(body, image, image.min())
    low, high = hu_range
    return np.where((image >= low) & (image <= high), image, 0).astype(np.float32)


def build_interpolator(
    case: Case, box: CropBox, pad: int
) -> sp.interpolate.RegularGridInterpolator:
    """Cubic interpolator of the CT in the crop frame, on the box padded by ``pad`` voxels."""
    start = time.perf_counter()
    lo = [max(0, a - pad) for a in box.start]
    hi = [min(n, b + pad) for b, n in zip(box.stop, case.image.shape, strict=True)]
    region = case.image[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    grids = tuple((np.arange(lo[a], hi[a]) - box.start[a]) * case.voxel_size[a] for a in range(3))
    interpolator = sp.interpolate.RegularGridInterpolator(
        grids,
        region,
        method="cubic",
        bounds_error=False,
        fill_value=None,
    )
    logger.debug("Interpolator built in %.2f s", time.perf_counter() - start)
    return interpolator


def prepare_case(case: Case, config: DetectionConfig) -> PreparedCase:
    """Crop the case and extract bone, as input for :func:`mspdetect.optimize.optimize`."""
    box = crop_box(case, config.crop)
    body = case.masks["Body"][box.slices] if "Body" in case.masks else None
    bone = bone_image(case.image[box.slices], config.hu_range, body)
    if not bone.any():
        raise ValueError(
            f"{case.case_id}: no voxels within hu_range {config.hu_range} in the cropped region",
        )
    R, t = array_frame(case.affine, box.start)
    return PreparedCase(
        bone=bone,
        interpolator=build_interpolator(case, box, config.crop.pad),
        voxel_size=case.voxel_size,
        crop=box,
        R=R,
        t=t,
    )
