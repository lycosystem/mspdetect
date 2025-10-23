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
