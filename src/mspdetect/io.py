"""Loading CT images and structure masks.

All volumes are reoriented to the closest canonical (RAS+) orientation on loading:
array axis 0 runs from the patient's left to right, axis 1 from posterior to anterior
and axis 2 from inferior to superior. The affine is kept, so every result can be
expressed in the scanner's world coordinates.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

NIFTI_EXTENSIONS = (".nii.gz", ".nii")


@dataclass(frozen=True)
class Case:
    """A CT image and its structure masks in canonical (RAS+) orientation.

    Parameters
    ----------
    case_id
        Identifier used in logs and result tables.
    image
        CT in Hounsfield units, shape ``(X, Y, Z)``.
    affine
        4x4 voxel-to-world (RAS+, mm) matrix.
    masks
        Boolean masks with the same shape as ``image``, keyed by structure name.

    """

    case_id: str
    image: np.ndarray
    affine: np.ndarray
    masks: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        image = np.asarray(self.image)
        affine = np.asarray(self.affine, dtype=float)
        if image.ndim != 3:
            raise ValueError(f"{self.case_id}: image must be 3D, got shape {image.shape}")
        if affine.shape != (4, 4):
            raise ValueError(f"{self.case_id}: affine must be 4x4, got {affine.shape}")
        masks = {}
        for name, mask in self.masks.items():
            mask = np.asarray(mask)
            if mask.shape != image.shape:
                raise ValueError(
                    f"{self.case_id}: mask '{name}' has shape {mask.shape}, "
                    f"image has {image.shape}",
                )
            masks[name] = mask.astype(bool, copy=False)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "affine", affine)
        object.__setattr__(self, "masks", masks)

    @property
    def voxel_size(self) -> np.ndarray:
        """Voxel size (mm) along the three array axes."""
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @property
    def voxel_volume(self) -> float:
        """Volume of one voxel in mm³."""
        return float(abs(np.linalg.det(self.affine[:3, :3])))

    def voxel_to_world(self, indices) -> np.ndarray:
        """World coordinates (mm) of voxel indices of shape ``(N, 3)``."""
        indices = np.asarray(indices, dtype=float)
        return indices @ self.affine[:3, :3].T + self.affine[:3, 3]


def _canonical(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(img)


def _check_same_grid(reference: nib.Nifti1Image, other: nib.Nifti1Image, name: str) -> None:
    if other.shape[:3] != reference.shape[:3]:
        raise ValueError(f"'{name}' has shape {other.shape}, the CT has {reference.shape}")
    if not np.allclose(other.affine, reference.affine, atol=1e-3):
        raise ValueError(
            f"'{name}' is not on the same voxel grid as the CT (affines differ):\n"
            f"{other.affine}\nvs\n{reference.affine}",
        )


def find_files(case_dir: str | Path, names: str | Sequence[str]) -> list[Path]:
    """NIfTI files below ``case_dir`` whose name (without extension) is one of ``names``.

    Matching is case-insensitive and searches sub-directories.
    """
    wanted = {names.lower()} if isinstance(names, str) else {n.lower() for n in names}
    found = []
    for root, _, files in os.walk(case_dir):
        for f in files:
            for ext in NIFTI_EXTENSIONS:
                if f.lower().endswith(ext) and f[: -len(ext)].lower() in wanted:
                    found.append(Path(root) / f)
                    break
    return sorted(found)


def _find_one(case_dir, names, what: str) -> Path | None:
    files = find_files(case_dir, names)
    if len(files) > 1:
        listed = "\n  ".join(str(f) for f in files)
        raise ValueError(f"Several files match {what} {names!r}:\n  {listed}")
    return files[0] if files else None


def load_case(
    case_dir: str | Path,
    image: str | Sequence[str] = "CT",
    structures: Mapping[str, str | Sequence[str]] | None = None,
    case_id: str | None = None,
) -> Case:
    """Load the CT and structure masks of one case from a directory of NIfTI files.

    Parameters
    ----------
    case_dir
        Directory with the case's NIfTI files (sub-directories are searched too).
    image
        File name of the CT without extension, or several alternatives.
    structures
        Structure name -> file name(s) without extension, e.g.
        ``{"GTVp": "ROI__[GTVp]", "Body": "ROI__[EXTERNAL]"}``. Structures without
        a matching file are skipped (with a log message); use ``case.masks`` to check.
    case_id
        Defaults to the directory name.

    Raises
    ------
    FileNotFoundError
        If no CT is found.
    ValueError
        If a name matches several files or a mask is not on the CT's voxel grid.

    """
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    image_path = _find_one(case_dir, image, "the image")
    if image_path is None:
        raise FileNotFoundError(f"No CT named {image!r} (.nii/.nii.gz) found in {case_dir}")

    ct = nib.load(image_path)
    masks = {}
    for name, file_names in (structures or {}).items():
        path = _find_one(case_dir, file_names, f"structure '{name}'")
        if path is None:
            logger.info("%s: structure '%s' not found", case_id or case_dir.name, name)
            continue
        mask_img = nib.load(path)
        _check_same_grid(ct, mask_img, name)
        masks[name] = np.asanyarray(_canonical(mask_img).dataobj) > 0.5

    ct = _canonical(ct)
    return Case(
        case_id=case_id or case_dir.name,
        image=ct.get_fdata(dtype=np.float32),
        affine=ct.affine,
        masks=masks,
    )
