"""
Synthetic phantoms with a known symmetry plane.

These tests check the geometric correctness and conventions, not accuracy on real
anatomy.
"""

import nibabel as nib
import numpy as np
import pytest

from mspdetect import Case, DetectionConfig, Plane

SHAPE = (64, 56, 30)
VOXEL_SIZE = (1.2, 1.2, 2.0)
ORIGIN = np.array([-38.0, -30.0, -25.0])
# Bone blobs (distance from plane, in-plane u, in-plane v) in mm; mirrored across the plane
BLOBS = [
    (15, -15, -12),
    (20, 10, -5),
    (10, 0, 12),
    (25, -5, 4),
    (0, 18, 0),
    (8, -20, 16),
]
BLOB_SIGMA = 3.5
# A GTV on the patient's left: centre 12 mm left of the plane, radius 6 mm
GTV_DISTANCE, GTV_RADIUS = -12.0, 6.0


def rotation(yaw_deg, roll_deg):
    yaw, roll = np.deg2rad([yaw_deg, roll_deg])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    ry = np.array([[np.cos(roll), 0, np.sin(roll)], [0, 1, 0], [-np.sin(roll), 0, np.cos(roll)]])
    return rz @ ry


def make_phantom(yaw_deg=4.0, roll_deg=3.0, shift_mm=2.5):
    """CT-like volume symmetric about a known plane, plus Body and GTVp masks (RAS)."""
    affine = np.eye(4)
    affine[:3, :3] = np.diag(VOXEL_SIZE)
    affine[:3, 3] = ORIGIN
    rot = rotation(yaw_deg, roll_deg)
    normal, e_u, e_v = rot[:, 0], rot[:, 1], rot[:, 2]
    centre = np.array([shift_mm, 0.0, 0.0])
    plane = Plane(normal, normal @ centre)

    idx = np.indices(SHAPE).reshape(3, -1).T
    world = idx @ affine[:3, :3].T + affine[:3, 3]
    s = world @ normal - plane.offset
    u, v = (world - centre) @ e_u, (world - centre) @ e_v

    image = np.full(len(idx), -1000.0)
    for a, bu, bv in BLOBS:
        r2 = (np.abs(s) - a) ** 2 + (u - bu) ** 2 + (v - bv) ** 2
        image += 2500 * np.exp(-r2 / (2 * BLOB_SIGMA**2))
    gtv = (s - GTV_DISTANCE) ** 2 + u**2 + v**2 <= GTV_RADIUS**2
    body = np.abs(s) < 35

    case = Case(
        "phantom",
        image.reshape(SHAPE).astype(np.float32),
        affine,
        {"GTVp": gtv.reshape(SHAPE), "Body": body.reshape(SHAPE)},
    )
    return case, plane


def flip_axes(data, affine, axes):
    """Same physical volume, stored with the given array axes reversed."""
    data, affine = data.copy(), affine.copy()
    for axis in axes:
        data = np.flip(data, axis=axis)
        affine[:3, 3] += affine[:3, axis] * (data.shape[axis] - 1)
        affine[:3, axis] *= -1
    return np.ascontiguousarray(data), affine


def write_case(directory, case, flip=()):
    """Save a case as NIfTI files CT.nii.gz, GTVp.nii.gz, Body.nii.gz."""
    directory.mkdir(parents=True, exist_ok=True)
    image, affine = flip_axes(case.image, case.affine, flip)
    nib.save(nib.Nifti1Image(image.astype(np.int16), affine), directory / "CT.nii.gz")
    for name, mask in case.masks.items():
        data, _ = flip_axes(mask, case.affine, flip)
        nib.save(nib.Nifti1Image(data.astype(np.uint8), affine), directory / f"{name}.nii.gz")
    return directory


WHOLE_SCAN = DetectionConfig(crop={"reference": None})


@pytest.fixture(scope="session")
def phantom():
    return make_phantom()


@pytest.fixture(scope="session")
def phantom_result(phantom):
    from mspdetect import detect_msp

    case, _ = phantom
    return detect_msp(case, WHOLE_SCAN)
