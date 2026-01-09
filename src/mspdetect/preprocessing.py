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


def _crop_structures_along_axis(struct_dict, axis, start, end):
    """Generic function to crop all structures along one axis given explicit bounds.

    Parameters
    ----------
    struct_dict : dict
        Dictionary of structures (arrays or (array, metadata) tuples)
    axis : int
        Axis along which to crop (0, 1, or 2)
    start : int
        Starting index (inclusive)
    end : int
        Ending index (exclusive, following numpy convention)

    Returns
    -------
    dict
        Cropped structures dictionary

    """
    struct_dict_cropped = {}
    idx = np.arange(start, end)

    for structure_name, structure_data in struct_dict.items():
        # Handle both (array, metadata) tuples and plain arrays
        if isinstance(structure_data, tuple):
            array = structure_data[0]
        else:
            array = structure_data

        cropped_array = np.take(array, idx, axis=axis)
        # Save the array
        struct_dict_cropped[structure_name] = cropped_array

    return struct_dict_cropped


def _adjust_for_mandible(struct_dict, axis, start, end):
    """Adjust bounds to include mandible if close to GTV."""
    mandible = struct_dict["Mandible"][0]
    mandible_slices = sp.ndimage.find_objects(mandible.astype(int))
    if mandible_slices:
        mandible_slice = mandible_slices[0][axis]
        mandible_start = mandible_slice.start
        if end <= mandible_start + 5:
            mandible_stop_lower = mandible_start + 5
            start = min(start, mandible_slice.start)
            end = max(end, mandible_stop_lower)
    return start, end


def _apply_margins(start, end, margins):
    """Apply margin(s) to crop bounds."""
    if isinstance(margins, int):
        lower_margin = upper_margin = margins
    elif isinstance(margins, tuple):
        lower_margin, upper_margin = margins
    else:
        raise ValueError(f"margins must be int or tuple, got {type(margins)}")
    return start - lower_margin, end + upper_margin


def _apply_fixed_size(start, end, fixed_size):
    """Apply fixed size window centered on reference."""
    ref_center = (start + end) // 2
    ref_extent = end - start

    # If reference larger than 2*fixed_size, use reference bounds
    if ref_extent <= 2 * fixed_size:
        start = ref_center - fixed_size
        end = ref_center + fixed_size
    return start, end


def _determine_z_axis_bounds(
    struct_dict,
    slice_axis=2,
    slice_range=None,
    margins=None,
    reference="GTVp",
    crop_mode="margins",
    fixed_size=None,
    exclude_mandible=False,
):
    """Determine crop bounds for primary (z) axis.

    Returns
    -------
    dict
        {"start": int, "end": int, "axis": int}

    """
    image = struct_dict["Image"][0]
    image_size = image.shape[slice_axis]

    if slice_range is not None:
        # Use explicit range, ignore margins and references
        start, end = slice_range
    else:
        # Use reference structure to determine range
        if reference in struct_dict:
            ref_struct = struct_dict[reference][0]
            slc = sp.ndimage.find_objects(ref_struct.astype(int))[0]
            start, end = slc[slice_axis].start, slc[slice_axis].stop - 1
        else:
            # Fall back to full image if reference not found
            start, end = 0, image_size - 1

        # Adjust for mandible (if needed)
        if not exclude_mandible and "Mandible" in struct_dict:
            start, end = _adjust_for_mandible(struct_dict, slice_axis, start, end)

        # Apply margins around reference structure if provided
        if crop_mode == "margins" and margins is not None:
            start, end = _apply_margins(start, end, margins)
        # Used fixed size around the GTVp
        elif crop_mode == "fixed_size" and fixed_size is not None:
            start, end = _apply_fixed_size(start, end, fixed_size)

    # Clamp to valid image bounds
    start = max(0, start)
    end = min(image_size - 1, end)

    if start > end:
        raise ValueError(f"Invalid crop range: start ({start}) > end ({end})")

    return {"start": start, "end": end, "axis": slice_axis}


def _determine_shoulder_bounds(
    struct_dict,
    slice_axis=1,
    max_extent_mm=200,
    reference="Body",
):
    """Determine crop bounds for shoulder (lateral) axis.

    Parameters
    ----------
    shoulder_axis : int
        Axis along which to crop shoulders (typically 0 or 1)
    max_extent_mm : float
        Maximum lateral extent to keep in mm

    Returns
    -------
    dict
        {"start": int, "end": int, "axis": int}

    """
    image = struct_dict["Image"][0]
    voxel_size = struct_dict["Image"][1][slice_axis]
    image_size = image.shape[slice_axis]

    # Find center from reference structure
    if reference in struct_dict:
        ref_struct = struct_dict[reference][0]
        slc = sp.ndimage.find_objects(ref_struct.astype(int))[0]
        center = (slc[slice_axis].start + slc[slice_axis].stop) // 2
    else:
        # Fallback to image center
        center = image_size // 2

    # Calculate extent in voxels
    max_extent_voxels = int(max_extent_mm / voxel_size)
    half_extent = max_extent_voxels // 2

    # Determine bounds centered on reference
    start = max(0, center - half_extent)
    end = min(image_size, center + half_extent)

    return {"start": start, "end": end, "axis": slice_axis}


def _create_interpolation_image(
    original_image,
    crop_bounds,
    interp_pad=10,
):
    """Create interpolation image with padding around crop bounds.

    Parameters
    ----------
    original_image : np.ndarray
        The original uncropped image
    crop_bounds : dict
        Dictionary with {"start": int, "end": int, "axis": int}
    interp_pad : int, default 10
        Number of slices to pad on each side for interpolation

    Returns
    -------
    tuple
        (interpolation_image, actual_pad_used)
        - interpolation_image : cropped image with padding
        - actual_pad_used : actual padding applied (may be less if truncated)

    """
    start = crop_bounds["start"]
    end = crop_bounds["end"]
    axis = crop_bounds["axis"]
    image_size = original_image.shape[axis]

    # Calculate padded bounds
    interp_start = max(0, start - interp_pad)
    interp_end = min(image_size - 1, end + interp_pad)

    # Calculate actual padding used
    actual_pad = min(start - interp_start, interp_pad)

    # Warn if padding was truncated at lower bound
    if interp_start == 0 and start < interp_pad:
        logger.warning(
            f"Interpolation padding truncated at lower bound "
            f"(requested {interp_pad}, got {actual_pad} slices)",
        )

    # Also check upper bound truncation
    if interp_end == image_size - 1 and (image_size - 1 - end) < interp_pad:
        upper_pad = image_size - 1 - end
        logger.warning(
            f"Interpolation padding truncated at upper bound "
            f"(requested {interp_pad}, got {upper_pad} slices)",
        )

    # Create padded interpolation image
    idx_interpolation = np.arange(interp_start, interp_end + 1)
    interpolation_image = np.take(original_image, idx_interpolation, axis=axis)

    return interpolation_image, actual_pad


def crop_patient_volumes(
    struct_dict,
    slice_axis=2,
    slice_range=None,
    margins=None,
    reference="GTVp",
    crop_mode="margins",
    fixed_size=None,
    exclude_mandible=False,
    crop_shoulders=None,
):
    """Crop patient volumes along primary and optional secondary axes."""
    original_image = struct_dict["Image"][0]

    # Determine and apply z-axis cropping around reference
    z_bounds = _determine_z_axis_bounds(
        struct_dict,
        slice_axis,
        slice_range,
        margins,
        reference,
        crop_mode,
        fixed_size,
        exclude_mandible,
    )

    struct_dict_cropped = _crop_structures_along_axis(
        struct_dict,
        z_bounds["axis"],
        z_bounds["start"],
        z_bounds["end"],
    )

    # z-axis interpolation image with padding
    interpolation_image, z_pad = _create_interpolation_image(
        original_image,
        z_bounds,
        interp_pad=10,
    )

    # collect crop information
    crop_info = {
        "z_axis": {**z_bounds, "pad": z_pad},
    }

    # Optional shoulder cropping
    if crop_shoulders is not None:
        shoulder_bounds = _determine_shoulder_bounds(
            struct_dict,
            crop_shoulders["axis"],
            crop_shoulders["max_extent_mm"],
            reference,
        )

        struct_dict_cropped = _crop_structures_along_axis(
            struct_dict_cropped,
            shoulder_bounds["axis"],
            shoulder_bounds["start"],
            shoulder_bounds["end"],
        )

        # add padding also for the shoulder bounds
        interpolation_image, shoulder_pad = _create_interpolation_image(
            interpolation_image,
            shoulder_bounds,
            interp_pad=10,
        )

        crop_info["shoulder_axis"] = {
            **shoulder_bounds,
            "pad": shoulder_pad,
        }

    return struct_dict_cropped, interpolation_image, crop_info


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
    crop_info,
    filename="interpolator.joblib",
    method="cubic",
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

    # Initialise physical-space sample grids without padding
    grids = [np.arange(image.shape[i]) * voxel_size[i] for i in range(3)]

    # Apply padding shifts for each padded axis
    for crop_key in crop_info:
        axis = crop_info[crop_key].get("axis", None)
        pad = crop_info[crop_key].get("pad", 0)
        # Shift so that first unpadded slice is at 0
        grids[axis] = (np.arange(image.shape[axis]) - pad) * voxel_size[axis]

    if os.path.exists(interpolator_path):
        interpolator = joblib.load(interpolator_path)
    else:
        start = time.time()
        interpolator = sp.interpolate.RegularGridInterpolator(
            tuple(grids),
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
