"""This module defines the geometry of the setup for detecting the MSP."""

import numpy as np


def vector_to_angles(vector):
    """Convert a 3D vector to spherical coordinates (azimuthal, polar, radius).

    Parameters
    ----------
    vector : array-like of shape (3,)
        The (x, y, z) vector.

    Returns
    -------
    np.ndarray
        Array of [azimuthal, polar, R] in radians and length.

    """
    x, y, z = vector

    # Calculate the distance from the origin
    R = np.linalg.norm(vector)

    # Calculate the distance from the origin in xy plane
    r = np.sqrt(x**2 + y**2)

    # Calculate the angle in the xy plane with respect to the x-axis
    azimuthal = np.arctan2(y, x)

    # Calculate the angle in the xz plane with respect to the z-axis
    polar = np.arctan2(r, z)

    return np.array([azimuthal, polar, R])


def angles_to_vector(azimuthal, polar, R):
    """Convert spherical coordinates (azimuthal, polar, radius) to a 3D vector.

    Parameters
    ----------
    azimuthal : float
        Azimuthal angle in radians.
    polar : float
        Polar angle in radians.
    R : float
        Radius (length).

    Returns
    -------
    np.ndarray
        The (x, y, z) vector.

    """
    x = R * np.sin(polar) * np.cos(azimuthal)
    y = R * np.sin(polar) * np.sin(azimuthal)
    z = R * np.cos(polar)

    return np.array([x, y, z])


def generate_normal(theta: float, phi: float) -> np.ndarray:
    """Generate a 3D normal vector from spherical angles theta and phi.

    Parameters
    ----------
    theta : float or np.ndarray
        Azimuthal angle(s) in radians.
    phi : float or np.ndarray
        Polar angle(s) in radians.

    Returns
    -------
    np.ndarray
        The normal vector(s) as (x, y, z).

    """
    # When theta and phi are arrays, we need to stack along the last axis
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Stack components along the last axis to get shape (..., 3)
    return np.stack([x, y, z], axis=-1)


def physical_plane_params(params, voxel_size):
    """Convert plane parameters from voxel space to physical space.

    :param params: Plane parameters in voxel space (theta, phi, L)
    :param voxel_size: np.array, voxel size
    """
    azimuthal, polar, L = params
    n = generate_normal(azimuthal, polar)
    vec_voxel = L * n
    vec_phys = np.multiply(vec_voxel, voxel_size)
    return vector_to_angles(vec_phys)


# don't need to convert to physical params, because the plane params are already the physical params
def crop_to_full_coords(params, crop_info, voxel_size):
    """Convert plane parameters from cropped to full image coordinates.

    Parameters
    ----------
    params : np.ndarray or tuple
        Plane parameters [azimuthal, polar, L] in cropped coordinates
    crop_info : dict
        Cropping axes and corresponding ranges
    voxel_size : tuple or np.ndarray
        Voxel spacing (sy, sx, sz) in mm

    Returns
    -------
    np.ndarray
        Plane parameters [azimuthal, polar, L] in full image coordinates

    """
    azimuthal, polar, L_cropped = params
    n = generate_normal(azimuthal, polar)

    # Map array axes to coordinate system
    axis_to_coord = {0: 1, 1: 0, 2: 2}

    # Accumulate offset from all cropped axes
    total_offset = np.zeros(3)
    for crop_key in crop_info:
        axis = crop_info[crop_key].get("axis")
        crop_start = crop_info[crop_key].get("start", 0)

        if axis is not None:
            coord_idx = axis_to_coord[axis]
            total_offset[coord_idx] += crop_start * voxel_size[axis]

    full_L = L_cropped + np.dot(total_offset, n)

    return np.array([azimuthal, polar, full_L])


def full_to_crop_coords(params, crop_info, voxel_size):
    """Convert plane parameters from full to cropped image coordinates.

    Parameters
    ----------
    params : np.ndarray or tuple
        Plane parameters [azimuthal, polar, L] in full image coordinates
    crop_info : dict
        Cropping axes and corresponding ranges
    voxel_size : tuple or np.ndarray
        Voxel spacing (sy, sx, sz) in mm

    Returns
    -------
    np.ndarray
        Plane parameters [azimuthal, polar, L] in cropped coordinates

    """
    azimuthal, polar, L_full = params
    n = generate_normal(azimuthal, polar)

    # Map array axes to coordinate system
    axis_to_coord = {0: 1, 1: 0, 2: 2}

    # Accumulate offset from all cropped axes
    total_offset = np.zeros(3)
    for crop_key in crop_info:
        axis = crop_info[crop_key].get("axis")
        crop_start = crop_info[crop_key].get("start", 0)

        if axis is not None:
            coord_idx = axis_to_coord[axis]
            total_offset[coord_idx] += crop_start * voxel_size[axis]

    cropped_L = L_full - np.dot(total_offset, n)

    return np.array([azimuthal, polar, cropped_L])
