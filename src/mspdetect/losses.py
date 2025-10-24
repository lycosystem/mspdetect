"""Loss functions for MSP detection."""

import numpy as np


def quadratic_loss(diff: np.ndarray) -> np.ndarray:
    """Compute the quadratic loss for an array of differences.

    Parameters
    ----------
    diff : np.ndarray
        Array of differences.

    Returns
    -------
    np.ndarray
        Quadratic loss values for each element in diff.

    """
    return 0.5 * diff**2


def huber_loss(diff: np.ndarray, delta: int = 300) -> np.ndarray:
    """Compute the Huber loss for an array of differences.

    Parameters
    ----------
    diff : np.ndarray
        Array of differences.
    delta : float, optional
        Huber loss delta parameter.

    Returns
    -------
    np.ndarray
        Huber loss values for each element in diff.

    """
    h_loss = np.where(
        np.abs(diff) <= delta,
        0.5 * diff**2,
        delta * (np.abs(diff) - 0.5 * delta),
    )
    return h_loss


def p_huber_loss(r: np.ndarray, c: int = 300.0, p: float = 1.4) -> np.ndarray:
    """Generalised Huber with polynomial tail."""
    a = np.abs(r)
    quad_mask = a <= c
    loss = np.empty_like(r, dtype=float)

    # quadratic core
    loss[quad_mask] = 0.5 * r[quad_mask] ** 2

    # tail: (2 c^{2-p} / (2-p)) |r|^p  - c^2 / (2-p)
    coeff = 2 * c ** (2 - p) / (2 - p)
    tail = coeff * (a[~quad_mask] ** p) - c**2 / (2 - p)
    loss[~quad_mask] = tail
    return loss


def welsch_loss(r: np.ndarray, c: int = 200) -> np.ndarray:
    """ρc(r) for Welsch."""
    factor = (r / c) ** 2
    return 0.5 * c**2 * (1.0 - np.exp(-factor))


def piecewise_loss(r: np.ndarray) -> np.ndarray:
    """Custom piecewise loss function:
    - Quadratic for |r| <= 200
    - Linear for 200 < |r| < 1500
    - Quadratic for |r| >= 1500
    Supports scalar or NumPy array input.

    Parameters
    ----------
    r : float or np.ndarray
        Residual value(s)

    Returns
    -------
    loss : float or np.ndarray
        Computed loss value(s)

    """
    r = np.asarray(r)
    abs_r = np.abs(r)

    loss = np.where(
        abs_r <= 200,
        r**2,
        np.where(
            abs_r < 1500,
            1700 * abs_r - 300_000,
            r**2,
        ),
    )
    return loss


def piecewise_intensity_loss(
    I_orig: np.ndarray,
    I_m: np.ndarray,
    threshold: int = 2800,
) -> np.ndarray:
    """Compute a continuous piecewise loss between original and mirrored intensities.

    For each voxel:
      - If I_m < threshold, loss = 0.5 * (I_orig - I_m)^2
      - Else, loss = 0.1 * (I_orig - I_m) + offset, chosen so that loss is continuous at I_m = threshold.

    Parameters
    ----------
    I_orig : array-like
        Original intensities.
    I_m : array-like
        Mirrored/interpolated intensities.
    threshold : float, optional
        Intensity threshold for switching between squared and linear loss.

    Returns
    -------
    loss : np.ndarray
        Array of the same shape as inputs, containing the piecewise continuous loss.

    """
    I_orig = np.asarray(I_orig)
    I_m = np.asarray(I_m)
    diff = I_orig - I_m

    # Compute diff at the threshold boundary for each voxel
    diff_thresh = I_orig - threshold

    # Squared part below threshold
    loss_sq = 0.5 * diff**2

    # Linear part above threshold, with offset for continuity
    a = 0.1
    offset = 0.5 * diff_thresh**2 - a * diff_thresh
    loss_lin = a * diff + offset

    # Combine piecewise
    loss = np.where(I_m < threshold, loss_sq, loss_lin)
    return loss
