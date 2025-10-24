"""Module defining functions for preprocessing the CT images."""

import os
import time

import joblib
import numpy as np
import scipy as sp
from loguru import logger


def mask_via_threshold(ct_image, HU_range=(300, 1500)):
    """Create a binary mask for the CT image using
    a specified Hounsfield Unit (HU) range.

    Parameters
    ----------
    ct_image : np.ndarray
        The CT image array.
    HU_range : tuple of (int, int)
        Lower and upper HU bounds for thresholding.

    Returns
    -------
    np.ndarray
        Binary mask where voxels within the HU range are set to 1.

    """
    bone_mask = np.zeros_like(ct_image)
    lower_bound, upper_bound = HU_range
    bone_mask[(ct_image >= lower_bound) & (ct_image <= upper_bound)] = 1

    return bone_mask


def crop_patient_volumes(struct_dict, slice_axis=2, slice_range=None):
    """Crop all structures in the patient dictionary along a specified axis and range.

    Parameters
    ----------
    struct_dict : dict
        Dictionary of structures, each as (image, voxel_size) or image array.
    slice_axis : int
        Axis along which to crop (default: 2, axial).
    slice_range : tuple or None
        (start, end) indices for cropping. If None, uses GTVp mask or full range.

    Returns
    -------
    dict
        Dictionary with cropped structure arrays.

    """
    image = struct_dict["Image"][0]
    if "GTVp" in struct_dict:
        gtvp = struct_dict["GTVp"][0]
        slc = sp.ndimage.find_objects(gtvp.astype(int))[0]
        start, end = slc[slice_axis].start, slc[slice_axis].stop - 1
    elif slice_range is not None:
        start, end = slice_range
    else:
        start, end = 0, image.shape[slice_axis] - 1

    if "Mandible" in struct_dict:
        mandible = struct_dict["Mandible"][0]
        mandible_slices = sp.ndimage.find_objects(mandible.astype(int))
        if mandible_slices:
            mandible_slice = mandible_slices[0][slice_axis]
            mandible_start = mandible_slice.start
            if end <= mandible_start + 5:
                mandible_stop_lower = mandible_start + 5
                start = min(start, mandible_slice.start)
                end = max(end, mandible_stop_lower)

    idx = np.arange(start, end + 1)

    def _crop(arr, idx):
        return np.take(arr, idx, axis=slice_axis)

    for structure in struct_dict:
        struct_dict[structure] = _crop(struct_dict[structure][0], idx)

    idx_interpolation = np.arange(start - 10, end + 1 + 10)
    interpolation_image = _crop(image, idx_interpolation)

    return struct_dict, interpolation_image


def preprocess_bone_image(struct_dict, HU_range):
    """Preprocess the CT image to extract bone regions
    using a HU threshold and optional body mask.

    Parameters
    ----------
    struct_dict : dict
        Dictionary of structures, must include 'Image' and optionally 'Body'.
    HU_range : tuple of (int, int)
        Lower and upper HU bounds for bone thresholding.

    Returns
    -------
    tuple
        proc_image : np.ndarray
            The processed CT image (optionally masked by 'Body').
        bone_ct : np.ndarray
            The CT image with only bone voxels retained.

    """
    proc_image = struct_dict["Image"].astype(np.int16)

    if "Body" in struct_dict:
        body = struct_dict["Body"]
        proc_image = np.where(body == 1, proc_image, np.min(proc_image))

    bone_mask = mask_via_threshold(proc_image, HU_range).astype(np.uint16)
    bone_ct = proc_image * bone_mask

    return proc_image, bone_ct


def get_cached_interpolator(
    output_dir,
    image,
    voxel_size,
    filename="interpolator.joblib",
    method="cubic",
    pad_slices=10,
):
    """Load or build and cache a RegularGridInterpolator for a cropped+Padded volume,
    shifting the z‐coordinates so that z=0 aligns with the first slice
    of the original crop.

    Parameters
    ----------
    output_dir : str
        Directory to look for/save the cached interpolator.
    image : np.ndarray
        Cropped+Padded image of shape (Ny, Nx, Nz_pad).
    voxel_size : sequence of 3 floats
        Physical spacing in mm along (Y, X, Z) axes.
    filename : str, optional
        Name of the .joblib file to load/save under output_dir.
    method : str, optional
        Interpolation method: 'cubic', 'linear', etc.
    pad_slices : int, optional
        Number of padded slices at the beginning of the volume. These
        will be assigned negative z-coordinates.

    Returns
    -------
    RegularGridInterpolator
        Interpolator mapping (y_mm, x_mm, z_mm_shifted) to intensity,
        where z_mm_shifted = 0 at the first slice *after* the pad.

    """
    os.makedirs(output_dir, exist_ok=True)
    interpolator_path = os.path.join(output_dir, filename)

    Ny, Nx, Nz_pad = image.shape
    dy, dx, dz = voxel_size

    # build physical‐space sample grids
    grid_y = np.arange(Ny) * dy
    grid_x = np.arange(Nx) * dx
    # shift z so that first un‐padded slice is at z=0
    grid_z = (np.arange(Nz_pad) - pad_slices) * dz

    if os.path.exists(interpolator_path):
        interpolator = joblib.load(interpolator_path)
    else:
        start = time.time()
        interpolator = sp.interpolate.RegularGridInterpolator(
            (grid_y, grid_x, grid_z),
            image,
            method=method,
            bounds_error=False,
            fill_value=None,
        )
        elapsed = time.time() - start
        logger.info(
            f"Interpolator built in {elapsed:.2f}s; cached to {interpolator_path}",
        )
        joblib.dump(interpolator, interpolator_path)

    return interpolator
