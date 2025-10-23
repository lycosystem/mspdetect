"""This module defines functions for visualization of various results."""

import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D


def plot_shifted_residuals_hist(
    diff,
    delta,
    folderpath=r"/home/loriskeller/Documents/Master Project/Results/28.07.25/test/4927494",
):
    """Plot the histogram of raw, median-shifted residuals and mark:
    • the Huber cutoff (δ̂) in magenta dashed lines,
    • the standard deviation (σ) of the fitted normal in blue dotted lines.

    Parameters
    ----------
    diff : array-like
        The raw residuals.
    delta : float
        The Huber loss threshold (δ̂).

    """
    # Center residuals
    shifted_diff = diff - np.median(diff)

    # Gaussian fit stats
    mu, std = shifted_diff.mean(), shifted_diff.std()

    # Histogram
    plt.figure(figsize=(7, 4))
    plt.hist(shifted_diff, bins=50, density=True, alpha=0.6, label="Shifted residuals")

    # Normal PDF
    x = np.linspace(shifted_diff.min(), shifted_diff.max(), 200)
    p = sp.stats.norm.pdf(x, mu, std)
    plt.plot(x, p, linewidth=2, label="Normal fit")

    # Huber threshold lines (magenta, single label)
    plt.axvline(delta, color="magenta", linestyle="--", label=f"δ̂ = {delta:.2f}")
    plt.axvline(-delta, color="magenta", linestyle="--")

    # Gaussian std lines (blue, single label)
    plt.axvline(std, color="blue", linestyle=":", label=f"σ = {std:.2f}")
    plt.axvline(-std, color="blue", linestyle=":")

    # Labels & legend
    plt.xlabel("Residuals r")
    plt.ylabel("Probability density of voxels")
    plt.legend()
    plt.tight_layout()
    plt.show()
    if folderpath is not None:
        path = os.path.join(folderpath, "shifted_residuals_histogram.pdf")
        plt.savefig(path, bbox_inches="tight")


def plot_mse_heatmap(
    azimuthal_angles: np.ndarray,
    polar_angles: np.ndarray,
    mse_array: np.ndarray,
    title: str = "MSE Heatmap",
    output_path: str = None,
    vmax_percentile: float = 99.0,
):
    """Plot and optionally save a heatmap of MSE values over a grid of angles,
    using LogNorm for logarithmic scaling of the colormap. Values above the
    percentile-based vmax are clipped to the same top color, while lower values
    are spread out for better differentiation.

    Parameters
    ----------
    azimuthal_angles : np.ndarray
        Array of azimuthal angles (radians).
    polar_angles : np.ndarray
        Array of polar angles (radians).
    mse_array : np.ndarray
        Flattened array of MSE values (length = P*T).
    title : str
        Title for the plot.
    output_path : str or None
        Directory to save the heatmap image (PDF). If None, displays interactively.
    vmax_percentile : float
        Percentile (0-100) of MSE to use as the maximum of the color scale.
        Values above this are clipped to the highest color.

    """
    # reshape and avoid zeros
    P = len(polar_angles)
    T = len(azimuthal_angles)
    mse_2d = mse_array.reshape(P, T)
    eps = np.finfo(float).eps
    mse_plot = mse_2d + eps

    # convert to degrees
    theta_deg = np.rad2deg(azimuthal_angles)
    phi_deg = np.rad2deg(polar_angles)
    TH, PH = np.meshgrid(theta_deg, phi_deg)

    # compute vmin and threshold-based vmax
    vmin = mse_plot.min()
    vmax = np.percentile(mse_plot, vmax_percentile)

    # setup LogNorm with clipping
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

    # plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(
        TH,
        PH,
        mse_plot,
        shading="auto",
        cmap="viridis",
        norm=norm,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("MSE log scale")

    ax.set_xlabel("Azimuthal angle θ (°)")
    ax.set_ylabel("Polar angle φ (°)")
    # ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.7)

    # save or show
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, "mse_heatmap.pdf")
        fig.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_robustness_heatmap(
    azimuthal_angles: np.ndarray,
    polar_angles: np.ndarray,
    error_array: np.ndarray,
    title: str = "Robustness Test Heatmap",
    output_path: str = None,
    vmax_percentile: float = 95.0,
):
    """Plot and save a heatmap of robustness test error metrics over a grid of initialization angles.

    Parameters
    ----------
    azimuthal_angles : np.ndarray
        Array of azimuthal angles (radians).
    polar_angles : np.ndarray
        Array of polar angles (radians).
    error_array : np.ndarray
        Flattened array of error metric values (length = P*T).
    title : str
        Title for the plot.
    output_path : str or None
        Directory to save the heatmap PDF. If None, displays interactively.
    vmax_percentile : float
        Percentile (0-100) of error values to use as the maximum of the color scale.
        Values above this are clipped to the highest color.

    """
    # Reshape error array to 2D grid
    P = len(polar_angles)
    T = len(azimuthal_angles)
    error_2d = error_array.reshape(P, T)

    # Handle infinite values by replacing with a large finite value
    finite_mask = np.isfinite(error_2d)
    if np.any(finite_mask):
        max_finite = np.max(error_2d[finite_mask])
        error_2d_plot = np.where(finite_mask, error_2d, max_finite * 2)
    else:
        error_2d_plot = np.ones_like(error_2d)

    # Convert angles to degrees for plotting
    theta_deg = np.rad2deg(azimuthal_angles)
    phi_deg = np.rad2deg(polar_angles)
    TH, PH = np.meshgrid(theta_deg, phi_deg)

    # Compute color scale limits
    if np.any(finite_mask):
        vmin = np.min(error_2d_plot[finite_mask])
        vmax = np.percentile(error_2d_plot[finite_mask], vmax_percentile)
    else:
        vmin, vmax = 0, 1

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a colormap that highlights low errors (good regions) in blue/green
    # and high errors (problematic regions) in red/yellow
    mesh = ax.pcolormesh(
        TH,
        PH,
        error_2d_plot,
        shading="auto",
        cmap="RdYlBu_r",  # Red-Yellow-Blue reversed (blue for low values)
        vmin=0,
        vmax=1,
    )

    # Add colorbar with proper labeling
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
    cbar.set_label("E", fontsize=12)

    # Mark any failed optimizations (infinite errors)
    if np.any(~finite_mask):
        failed_theta = TH[~finite_mask]
        failed_phi = PH[~finite_mask]
        ax.scatter(
            failed_theta,
            failed_phi,
            c="white",
            s=50,
            marker="x",
            linewidths=2,
            label="Failed optimizations",
        )
        ax.legend()

    # Formatting
    ax.set_xlabel("Azimuthal angle θ (°)", fontsize=16)
    ax.set_ylabel("Polar angle φ (°)", fontsize=16)
    # ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.3)

    # Save or show
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, "robustness_heatmap.pdf")
        fig.savefig(out_file, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Robustness heatmap saved to: {out_file}")
        plt.close(fig)
    else:
        plt.show()


def load_and_plot_robustness_results(
    results_path,
    title="Robustness Test Heatmap",
    vmax_percentile=95.0,
):
    """Load saved robustness test results and regenerate the heatmap.

    Parameters
    ----------
    results_path : str
        Path to the directory containing saved robustness test results
    title : str, optional
        Title for the regenerated heatmap
    vmax_percentile : float, optional
        Percentile for color scale maximum

    Returns
    -------
    dict
        Loaded results dictionary

    """
    import json

    # Load the compressed numpy results
    results_file = os.path.join(results_path, "robustness_test_results.npz")
    metadata_file = os.path.join(results_path, "robustness_test_metadata.json")

    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    print(f"Loading robustness test results from {results_path}")

    # Load numpy data
    data = np.load(results_file)

    # Load metadata
    with open(metadata_file) as f:
        metadata = json.load(f)

    # Reconstruct results dictionary
    results = {
        "error_metrics": data["error_metrics"],
        "azimuthal_angles": data["azimuthal_angles"],
        "polar_angles": data["polar_angles"],
        "theta_grid": data["theta_grid"],
        "phi_grid": data["phi_grid"],
        "L_grid": data["L_grid"],
        "optimal_plane_params": data["optimal_plane_params"],
        "initial_params": data["initial_params"],
        "optimized_params": data["optimized_params"],
        "initial_objectives": data["initial_objectives"],
        "optimized_objectives": data["optimized_objectives"],
        "deltas": data["deltas"],
        "success_flags": data["success_flags"],
        "nit_counts": data["nit_counts"],
        "metadata": metadata,
    }

    # Regenerate the heatmap
    print("Regenerating robustness heatmap...")
    plot_robustness_heatmap(
        azimuthal_angles=results["azimuthal_angles"],
        polar_angles=results["polar_angles"],
        error_array=results["error_metrics"],
        title=title,
        output_path=results_path,
        vmax_percentile=vmax_percentile,
    )

    return results


def make_plane_gif(
    image_3d,
    voxel_size,
    plane_params,
    objective_values,
    output_path,
    duration=0.001,
):
    """Create a GIF showing the evolution of plane contours on the middle slice of a 3D image.

    Parameters
    ----------
    image_3d : np.ndarray
        The 3D image array (coronal, sagittal, axial).
    voxel_size : tuple of float
        Spacing in mm along (coronal=y, sagittal=x, axial=z).
    plane_params : np.ndarray
        Array of plane parameters for each frame (N, 3).
    objective_values : np.ndarray
        Array of objective values for each frame (N,).
    output_path : str
        Directory to save the resulting GIF.
    duration : float, optional
        Time in seconds each frame is shown.

    """
    H, W, D = image_3d.shape
    sy, sx, sz = voxel_size

    # middle slice and its physical z
    z0 = D // 2
    z0_mm = z0 * sz
    slice_img = image_3d[:, :, z0]

    # physical grid for contour
    y_mm = np.arange(H) * sy
    x_mm = np.arange(W) * sx
    X_mm, Y_mm = np.meshgrid(
        x_mm,
        y_mm,
    )  # note: meshgrid(x,y) -> X shape (len(y), len(x))

    writer = imageio.get_writer(
        os.path.join(output_path, "plane_optimization.gif"),
        mode="I",
        duration=duration,
        loop=0,
    )

    for idx in range(plane_params.shape[0]):
        # extract parameters
        theta, phi, L_mm = plane_params[idx]
        objective_value = objective_values[idx]
        # compute plane normal
        n = np.array(
            [
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ],
        )
        nx, ny, nz = n

        # plane constant
        C = L_mm - nz * z0_mm

        # contour function in mm
        F = nx * X_mm + ny * Y_mm - C

        # plot
        fig = plt.figure(figsize=(6, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(
            slice_img,
            cmap="gray",
            origin="upper",
            extent=[x_mm[0], x_mm[-1], y_mm[-1], y_mm[0]],
        )
        ax.contour(X_mm, Y_mm, F, levels=[0], colors="red")
        ax.set_title(
            f"Plane {idx + 1}: θ={np.rad2deg(theta):.1f}°, "
            f"φ={np.rad2deg(phi):.1f}°, L={L_mm:.1f} mm\nMSE={objective_value:.2f}",
        )
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        # ax.axis('off')

        # render to image
        canvas.draw()
        buf, (w, h) = canvas.print_to_buffer()
        frame = np.frombuffer(buf, dtype="uint8").reshape(h, w, 4)[..., :3]
        writer.append_data(frame)

        plt.close(fig)

    writer.close()
    print(f"Saved GIF to {output_path}")


def make_optimization_gif(
    objective_values,
    output_path,
    duration=0.001,
    title="Optimization Progress",
):
    """Create a GIF showing the evolution of the objective function during optimization.
    Each frame shows all previous objective values as connected lines, creating a
    cumulative visualization of the optimization process.

    Parameters
    ----------
    objective_values : np.ndarray or list
        Array of objective values for each optimization iteration.
    output_path : str
        Directory to save the resulting GIF.
    duration : float, optional
        Time in seconds each frame is shown.
    title : str, optional
        Title for the plot.

    """
    objective_values = np.array(objective_values)
    n_iterations = len(objective_values)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Set up the writer with infinite loop for PowerPoint presentations
    writer = imageio.get_writer(
        os.path.join(output_path, "optimization_progress.gif"),
        mode="I",
        duration=duration,
        loop=0,
    )

    # Determine plot limits
    y_min = np.min(objective_values) * 0.95
    y_max = np.max(objective_values) * 1.05

    # Handle case where all values are the same
    if y_min == y_max:
        y_range = abs(y_min) * 0.1 if y_min != 0 else 1.0
        y_min -= y_range
        y_max += y_range

    for frame_idx in range(1, n_iterations + 1):
        # Create figure for this frame
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get data up to current frame
        x_data = np.arange(1, frame_idx + 1)
        y_data = objective_values[:frame_idx]

        # Plot all previous points as connected line
        ax.plot(x_data, y_data, "b-", linewidth=2, marker="o", markersize=4, alpha=0.7)

        # Highlight the current point
        if frame_idx > 0:
            ax.plot(
                frame_idx,
                objective_values[frame_idx - 1],
                "ro",
                markersize=8,
                markerfacecolor="red",
                markeredgecolor="darkred",
                markeredgewidth=2,
            )

        # Set plot properties
        ax.set_xlim(0, n_iterations + 1)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Objective Function Value", fontsize=12)
        ax.set_title(
            f"{title} - Iteration {frame_idx}/{n_iterations}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Add current value as text
        current_value = objective_values[frame_idx - 1]
        ax.text(
            0.02,
            0.98,
            f"Current Value: {current_value:.2e}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Add best (minimum) value so far
        best_value = np.min(y_data)
        best_idx = np.argmin(y_data) + 1
        ax.text(
            0.02,
            0.88,
            f"Best Value: {best_value:.2e} (Iter {best_idx})",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        )

        # Set y-axis to log scale if values span multiple orders of magnitude
        if y_max / y_min > 100 and y_min > 0:
            ax.set_yscale("log")
            ax.set_ylabel("Objective Function Value (log scale)", fontsize=12)

        plt.tight_layout()

        # Convert plot to image array
        canvas = fig.canvas
        canvas.draw()
        buf, (w, h) = canvas.print_to_buffer()
        frame = np.frombuffer(buf, dtype="uint8").reshape(h, w, 4)[..., :3]

        # Add frame to GIF
        writer.append_data(frame)

        plt.close(fig)

    writer.close()
    print(
        f"Saved optimization GIF to {os.path.join(output_path, 'optimization_progress.gif')}",
    )


def plot_axial_coronal(
    struct_dict: dict,
    voxel_size: tuple,
    z_index: int,
    y_line: int,
    save_path: str,
    plane_coeffs_list: list = None,
    optimization_methods_list: list = None,
    axial_crop: tuple = None,
    coronal_crop: tuple = None,
    crop_size: int = None,
    figsize: tuple = (12, 6),
):
    """Plot axial and coronal views side by side and save as PDF, with adjustable cropping.

    Parameters
    ----------
    struct_dict : dict
        Must contain 'Image' (ny,nx,nz array) and any mask arrays.
    voxel_size : tuple
        (sy, sx, sz) spacing in mm along Y, X, Z axes.
    z_index : int
        Axial slice index.
    y_line : int
        Y-index for coronal view.
    save_path : str
        Path to output PDF.
    plane_coeffs_list : list of (theta, phi, L), optional
        Planes to overlay.
    optimization_methods_list : list of str, optional
        Labels for plane overlays.
    axial_crop : (x_min, x_max, y_min, y_max), optional
        Pixel indices to crop axial view: X lower/upper, Y lower/upper.
        If None and crop_size is provided, uses center of mass based cropping.
    coronal_crop : (x_min, x_max, z_min, z_max), optional
        Pixel indices to crop coronal view: X lower/upper, Z lower/upper.
        If None and crop_size is provided, uses center of mass based cropping.
    crop_size : int, optional
        Half-width in pixels around center of mass of Body mask for automatic cropping.
        Only used if axial_crop/coronal_crop are None and 'Body' is in struct_dict.
    figsize : (w,h)
        Figure size in inches.

    """
    if plane_coeffs_list is None:
        plane_coeffs_list = []
    if optimization_methods_list is None:
        optimization_methods_list = []

    def _arr(obj):
        return obj[0] if isinstance(obj, (tuple, list)) else obj

    # Load image
    image = _arr(struct_dict["Image"])
    H, W, D = image.shape
    sy, sx, sz = voxel_size

    # Center of mass based cropping if crop_size is provided and Body mask exists
    if (
        crop_size
        and "Body" in struct_dict
        and (axial_crop is None or coronal_crop is None)
    ):
        body = _arr(struct_dict["Body"])
        # Compute center of mass for body mask
        sum_y = body.sum(axis=(1, 2))
        sum_x = body.sum(axis=(0, 2))
        sum_z = body.sum(axis=(0, 1))
        cy = int(round((np.arange(H) * sum_y).sum() / (sum_y.sum() or 1)))
        cx = int(round((np.arange(W) * sum_x).sum() / (sum_x.sum() or 1)))
        cz = int(round((np.arange(D) * sum_z).sum() / (sum_z.sum() or 1)))

        # Auto-generate crop bounds around center of mass
        auto_axial_crop = (
            max(0, cx - crop_size),
            min(W, cx + crop_size),
            max(0, cy - crop_size),
            min(H, cy + crop_size),
        )
        auto_coronal_crop = (
            max(0, cx - crop_size),
            min(W, cx + crop_size),
            max(0, cz - crop_size),
            min(D, cz + crop_size),
        )
    else:
        auto_axial_crop = None
        auto_coronal_crop = None

    # Default full extents or use provided/auto-generated crops
    ax_x0, ax_x1, ax_y0, ax_y1 = 0, W, 0, H
    if axial_crop:
        ax_x0, ax_x1, ax_y0, ax_y1 = axial_crop
        ax_x0, ax_x1 = max(0, ax_x0), min(W, ax_x1)
        ax_y0, ax_y1 = max(0, ax_y0), min(H, ax_y1)
    elif auto_axial_crop:
        ax_x0, ax_x1, ax_y0, ax_y1 = auto_axial_crop
        ax_x0, ax_x1 = max(0, ax_x0), min(W, ax_x1)
        ax_y0, ax_y1 = max(0, ax_y0), min(H, ax_y1)

    cor_x0, cor_x1, cor_z0, cor_z1 = 0, W, 0, D
    if coronal_crop:
        cor_x0, cor_x1, cor_z0, cor_z1 = coronal_crop
        cor_x0, cor_x1 = max(0, cor_x0), min(W, cor_x1)
        cor_z0, cor_z1 = max(0, cor_z0), min(D, cor_z1)
    elif auto_coronal_crop:
        cor_x0, cor_x1, cor_z0, cor_z1 = auto_coronal_crop
        cor_x0, cor_x1 = max(0, cor_x0), min(W, cor_x1)
        cor_z0, cor_z1 = max(0, cor_z0), min(D, cor_z1)

    # Crop arrays
    ax_img = image[ax_y0:ax_y1, ax_x0:ax_x1, :]
    cor_img = image[:, cor_x0:cor_x1, cor_z0:cor_z1]

    # Physical coords
    x_ax = np.arange(ax_x0, ax_x1) * sx
    y_ax = np.arange(ax_y0, ax_y1) * sy
    z_ax = np.arange(0, D) * sz
    X_ax, Y_ax = np.meshgrid(x_ax, y_ax)

    x_cor = np.arange(cor_x0, cor_x1) * sx
    z_cor = np.arange(cor_z0, cor_z1) * sz
    X_cor, Z_cor = np.meshgrid(x_cor, z_cor)

    # Setup figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # ----- Axial -----
    ax0.imshow(
        ax_img[:, :, z_index],
        cmap="gray",
        extent=[x_ax[0], x_ax[-1], y_ax[-1], y_ax[0]],
    )
    # ax0.set_title(f'Axial z={z_index}')
    ax0.set_xlabel("X (mm)")
    ax0.set_ylabel("Y (mm)")
    # horizontal guide at Y = y_line
    y0 = y_ax[y_line - ax_y0]  # adjust for cropping
    ax0.axhline(y=y0, color="yellow", linestyle="--")

    # Overlay contours
    struct_names = [k for k in struct_dict if k != "Image"]
    cmap10 = plt.cm.get_cmap("tab10").colors
    col = {n: cmap10[i % 10] for i, n in enumerate(struct_names)}
    for name in struct_names:
        mask = _arr(struct_dict[name])[ax_y0:ax_y1, ax_x0:ax_x1, :]
        sl = mask[:, :, z_index]
        if sl.any():
            ax0.contour(X_ax, Y_ax, sl, levels=[0.5], colors=[col[name]], linewidths=1)

    # Planes
    plane_cols = ["red", "purple", "cyan"]
    for i, (theta, phi, L) in enumerate(plane_coeffs_list):
        nx = np.sin(phi) * np.cos(theta)
        ny = np.sin(phi) * np.sin(theta)
        nz = np.cos(phi)
        C = L - nz * (z_index * sz)
        F = nx * X_ax + ny * Y_ax - C
        ax0.contour(
            X_ax,
            Y_ax,
            F,
            levels=[0],
            colors=[plane_cols[i % len(plane_cols)]],
            linewidths=1,
        )

    # ----- Coronal -----
    cor_slice = cor_img[x_slice_index if False else y_line, :, :]  # ignore bug
    cor_slice = cor_img[y_line, :, :].T
    ax1.imshow(
        cor_slice,
        cmap="gray",
        origin="lower",
        extent=[x_cor[0], x_cor[-1], z_cor[0], z_cor[-1]],
    )
    # ax1.set_title(f'Coronal y={y_line}')
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Z (mm)")
    # Guide line at current Z slice
    z0 = z_ax[z_index - cor_z0]  # adjust for cropping
    ax1.axhline(y=z0, color="yellow", linestyle="--")

    for name in struct_names:
        mask = _arr(struct_dict[name])
        m2 = mask[y_line, cor_x0:cor_x1, cor_z0:cor_z1].T
        if m2.any():
            ax1.contour(
                X_cor,
                Z_cor,
                m2,
                levels=[0.5],
                colors=[col[name]],
                linewidths=1,
            )

    for i, (theta, phi, L) in enumerate(plane_coeffs_list):
        nx = np.sin(phi) * np.cos(theta)
        ny = np.sin(phi) * np.sin(theta)
        nz = np.cos(phi)
        y_phys = y_line * sy
        C = L
        F2 = nx * X_cor + ny * y_phys + nz * Z_cor - L
        ax1.contour(
            X_cor,
            Z_cor,
            F2,
            levels=[0],
            colors=[plane_cols[i % len(plane_cols)]],
            linewidths=1,
        )

    # Legend
    hds = [Line2D([0], [0], color=col[n], lw=2, label=n) for n in struct_names]
    hds += [
        Line2D(
            [0],
            [0],
            color=plane_cols[i],
            lw=2,
            label=optimization_methods_list[i]
            if i < len(optimization_methods_list)
            else f"Plane{i}",
        )
        for i in range(len(plane_coeffs_list))
    ]
    ax0.legend(handles=hds, loc="upper right", fontsize=11, frameon=True)

    plt.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_3d_slice_as_vector(array3d, save_dir, index, factor, filename):
    """Save 2D slices from all three orthogonal directions of a 3D array as high-quality PNG images.

    Parameters
    ----------
    array3d : numpy.ndarray
        3D array of shape (H, W, D) where H=height(y), W=width(x), D=depth(z).
    save_dir : str
        Directory where the PNG images will be saved.
    index : int or dict
        If int: uses same index for all directions.
        If dict: should contain keys 'axial', 'coronal', 'sagittal' with respective slice indices.
    factor : int
        Number of pixels to crop from each border (both top/bottom and left/right).
    filename : str
        Base filename (will be suffixed with direction names).

    """
    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    H, W, D = array3d.shape

    # Handle index parameter
    if isinstance(index, int):
        # Use same index for all directions, but validate against each dimension
        indices = {
            "axial": min(index, D - 1),  # z-direction slice
            "coronal": min(index, H - 1),  # y-direction slice
            "sagittal": min(index, W - 1),  # x-direction slice
        }
    elif isinstance(index, dict):
        indices = index.copy()
        # Validate each index
        if "axial" in indices and not (0 <= indices["axial"] < D):
            raise ValueError(
                f"Axial index {indices['axial']} is out of bounds for depth {D}.",
            )
        if "coronal" in indices and not (0 <= indices["coronal"] < H):
            raise ValueError(
                f"Coronal index {indices['coronal']} is out of bounds for height {H}.",
            )
        if "sagittal" in indices and not (0 <= indices["sagittal"] < W):
            raise ValueError(
                f"Sagittal index {indices['sagittal']} is out of bounds for width {W}.",
            )
    else:
        raise ValueError(
            "Index must be an integer or a dictionary with direction keys.",
        )

    # Define slice extraction for each direction
    slice_configs = {
        "axial": {
            "slice": array3d[:, :, indices.get("axial", D // 2)]
            if "axial" in indices or isinstance(index, int)
            else None,
            "description": f"Axial (z={indices.get('axial', D // 2)})",
        },
        "coronal": {
            "slice": array3d[indices.get("coronal", H // 2), :, :]
            if "coronal" in indices or isinstance(index, int)
            else None,
            "description": f"Coronal (y={indices.get('coronal', H // 2)})",
        },
        "sagittal": {
            "slice": array3d[:, indices.get("sagittal", W // 2), :]
            if "sagittal" in indices or isinstance(index, int)
            else None,
            "description": f"Sagittal (x={indices.get('sagittal', W // 2)})",
        },
    }

    # Save each direction
    for direction, config in slice_configs.items():
        if config["slice"] is None:
            continue

        slice_2d = config["slice"]

        # Crop borders by factor if needed
        if (
            factor > 0
            and slice_2d.shape[0] > 2 * factor
            and slice_2d.shape[1] > 2 * factor
        ):
            slice_2d = slice_2d[factor:-factor, factor:-factor]

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(slice_2d, cmap="gray", interpolation="none", aspect="equal")
        ax.axis("off")  # Hide axes for a clean image

        # Adjust layout to remove padding/margins
        fig.tight_layout(pad=0)

        # Define output filepath
        filepath = os.path.join(save_dir, f"{filename}_{direction}.png")

        # Save as high-quality PNG
        fig.savefig(filepath, format="png", bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig)

        print(f"Saved {config['description']} slice to {filepath}")

    print(f"Saved all orthogonal slices for {filename} to {save_dir}")


# ─── Master mapping: every structure → specific color ─────────────────────────
STRUCTURE_COLOR_MAP = {
    "GTV-T": "tab:orange",
    "Mandible": "tab:pink",
    "Body": "tab:blue",
    "Spinal Cord": "tab:green",
}
DEFAULT_STRUCT_COLOR = "purple"


def plot_and_save_axial_slice(
    struct_dict: dict,
    voxel_size: tuple,
    z_indices,
    save_path: str,
    crop_size: int = 50,
    plane_coeffs_list: list = None,
    optimization_methods_list: list = None,
    figsize_per: tuple = (6, 6),
):
    """Plot one or multiple axial slices. If `z_indices` is a list or tuple, a single figure
    with side-by-side subplots is created. If an integer, only that slice is plotted.

    Parameters
    ----------
    struct_dict : dict
        Must include 'Image' (ny,nx,nz array) and any structure masks keyed by name.
    voxel_size : tuple
        (sy, sx, sz) spacing in mm for Y, X, Z axes.
    z_indices : int or sequence of int
        One or multiple axial slice indices.
    save_path : str
        Output file path (including .pdf).
    crop_size : int, optional
        Half-window in voxels around the body center-of-mass for cropping. Default 50.
    plane_coeffs_list : list of (theta, phi, L), optional
        Planes to overlay on each slice.
    optimization_methods_list : list of str, optional
        Legend labels for the plane overlays.
    figsize_per : (width, height), optional
        Size in inches for each subplot.

    """
    # Normalize z_indices
    if isinstance(z_indices, (int, np.integer)):
        z_list = [z_indices]
    else:
        z_list = list(z_indices)
    n = len(z_list)

    # Defaults
    if plane_coeffs_list is None:
        plane_coeffs_list = []
    if optimization_methods_list is None:
        optimization_methods_list = []

    def _arr(obj):
        return obj[0] if isinstance(obj, (tuple, list)) else obj

    # Load image and optional body
    image = _arr(struct_dict["Image"])
    ny, nx, nz = image.shape
    sy, sx, sz = voxel_size
    body = _arr(struct_dict.get("Body", image > 0))

    # Compute crop bounds
    if crop_size and crop_size > 0:
        sum_y = body.sum(axis=(1, 2))
        sum_x = body.sum(axis=(0, 2))
        cy = int(round((np.arange(ny) * sum_y).sum() / (sum_y.sum() or 1)))
        cx = int(round((np.arange(nx) * sum_x).sum() / (sum_x.sum() or 1)))
        y0, y1 = max(0, cy - crop_size), min(ny, cy + crop_size)
        x0, x1 = max(0, cx - crop_size), min(nx, cx + crop_size)
    else:
        y0, y1, x0, x1 = 0, ny, 0, nx

    # Crop data
    img_c = image[y0:y1, x0:x1, :]

    # Physical coordinates for plotting
    y_mm = np.arange(y0, y1) * sy
    x_mm = np.arange(x0, x1) * sx
    Xg, Yg = np.meshgrid(x_mm, y_mm)

    # Prepare figure and axes
    fig_w, fig_h = figsize_per[0] * n, figsize_per[1]
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), constrained_layout=True)
    if n == 1:
        axes = [axes]

    # Precompute structure color mapping
    struct_names = [name for name in struct_dict if name != "Image"]
    cmap10 = plt.cm.get_cmap("tab10").colors
    struct_colors = {
        name: cmap10[i % len(cmap10)] for i, name in enumerate(struct_names)
    }

    # Loop through each slice index
    for ax, z in zip(axes, z_list, strict=False):
        # Show image
        ax.imshow(
            img_c[:, :, z],
            cmap="gray",
            extent=[x_mm[0], x_mm[-1], y_mm[-1], y_mm[0]],
        )
        # Overlay structure contours
        for name in struct_names:
            mask = _arr(struct_dict[name])
            mc = mask[y0:y1, x0:x1, :]
            sl = mc[:, :, z]
            if sl.any():
                ax.contour(
                    Xg,
                    Yg,
                    sl,
                    levels=[0.5],
                    colors=[struct_colors[name]],
                    linewidths=3,
                )
        # Overlay plane contours
        # plane_colors = ['red','purple','cyan','lime','magenta']
        plane_colors = ["red"]
        for i, (theta, phi, L) in enumerate(plane_coeffs_list):
            nx_f = np.sin(phi) * np.cos(theta)
            ny_f = np.sin(phi) * np.sin(theta)
            nz_f = np.cos(phi)
            C = L - nz_f * (z * sz)
            F = nx_f * Xg + ny_f * Yg - C
            ax.contour(
                Xg,
                Yg,
                F,
                levels=[0],
                colors=[plane_colors[i % len(plane_colors)]],
                linewidths=1.5,
            )
        # Fix aspect and remove axes
        ax.set_xlim(x_mm[0], x_mm[-1])
        ax.set_ylim(y_mm[-1], y_mm[0])
        ax.set_aspect("equal")
        ax.axis("off")
        # ax.set_title(f'Slice {z}', fontsize=12)

    # # Build legend on first axis
    # handles = [Line2D([0],[0], color=struct_colors[n], lw=2, label=n)
    #            for n in struct_names]

    handles = [
        Line2D(
            [0],
            [0],
            color=struct_colors[n],
            lw=2,
            label=("Bone" if n == "Mandible" else n),
        )
        for n in struct_names
    ]
    for i, method in enumerate(optimization_methods_list):
        c = plane_colors[i % len(plane_colors)]
        handles.append(Line2D([0], [0], color=c, lw=2, label=method))
    axes[0].legend(handles=handles, loc="upper right", fontsize=11, frameon=True)

    # Save figure
    fig.savefig(save_path, format="png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


import numpy as np

# Color for GTV-T contour positions
POSITION_COLORS = ["tab:blue", "tab:orange", "tab:green"]  # for shifts [max, mid, 0]
PLANE_COLOR = "red"  # explicit plane color


def plot_gtvt_positions(
    struct_dict: dict,
    voxel_size: tuple,
    z_index: int,
    plane_coeffs: tuple,
    save_path: str,
    shifts: list = None,
    crop_size: int = None,
    figsize_per: tuple = (5, 5),
):
    """Plot three side-by-side axial slices at slice index `z_index`, each showing the
    GTV-T contour shifted along the negative XY-plane projection of the provided
    plane normal. The leftmost is the largest shift, the rightmost is no shift.
    Also overlays the plane intersection line (midline) on each subplot, clipped to
    the cropped image bounds.

    Parameters
    ----------
    struct_dict : dict
        Must contain 'Image' (ny,nx,nz array) and 'GTV-T' mask.
        Optionally 'Body' for cropping around center of mass.
    voxel_size : tuple
        (sy, sx, sz) voxel spacing in mm.
    z_index : int
        Axial slice index to display.
    plane_coeffs : tuple (theta, phi, L)
        Plane normal spherical coords (radians) and offset L in mm.
    save_path : str
        Output PDF file path.
    shifts : list of int, optional
        Voxel offsets along negative normal for the three positions. Default [20,10,0].
    crop_size : int, optional
        Half-window (voxels) around body COM to crop slice. If None, no crop.
    figsize_per : (w,h), optional
        Figure size (inches) per subplot.

    """

    def _arr(x):
        return x[0] if isinstance(x, (tuple, list)) else x

    # Default shifts
    if shifts is None:
        shifts = [20, 10, 0]
    if len(shifts) != 3:
        raise ValueError("shifts must be a list of three values")

    # Load arrays
    img = _arr(struct_dict["Image"])
    gtv = _arr(struct_dict["GTV-T"])
    ny, nx, nz = img.shape
    sy, sx, sz = voxel_size

    # Determine crop bounds
    if crop_size and "Body" in struct_dict:
        body = _arr(struct_dict["Body"])
        sum_y = body.sum(axis=(1, 2))
        sum_x = body.sum(axis=(0, 2))
        cy = int(round((np.arange(ny) * sum_y).sum() / (sum_y.sum() or 1)))
        cx = int(round((np.arange(nx) * sum_x).sum() / (sum_x.sum() or 1)))
        y0, y1 = max(0, cy - crop_size), min(ny, cy + crop_size)
        x0, x1 = max(0, cx - crop_size), min(nx, cx + crop_size)
    else:
        y0, y1, x0, x1 = 0, ny, 0, nx

    # Crop data
    img_c = img[y0:y1, x0:x1, :]
    gtv_c = gtv[y0:y1, x0:x1, :]

    # Physical coordinates
    y_mm = np.arange(y0, y1) * sy
    x_mm = np.arange(x0, x1) * sx
    Xg, Yg = np.meshgrid(x_mm, y_mm)

    # Plane normal components
    theta, phi, L = plane_coeffs
    nx_f = np.sin(phi) * np.cos(theta)
    ny_f = np.sin(phi) * np.sin(theta)
    nz_f = np.cos(phi)
    # XY unit vector along plane normal projection
    norm_xy = np.hypot(nx_f, ny_f)
    ux = nx_f / norm_xy if norm_xy > 0 else 0
    uy = ny_f / norm_xy if norm_xy > 0 else 0

    # Midline constant
    z_mm = z_index * sz
    C = L - nz_f * z_mm

    # Clipping bounds in mm
    x_min, x_max = x_mm.min(), x_mm.max()
    y_min, y_max = y_mm.min(), y_mm.max()

    # Subplots
    n = 3
    fig_w, fig_h = figsize_per[0] * n, figsize_per[1]
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), constrained_layout=True)

    for ax, shift, color in zip(axes, shifts, POSITION_COLORS, strict=False):
        # show image
        ax.imshow(
            img_c[:, :, z_index],
            cmap="gray",
            extent=[x_min, x_max, y_max, y_min],
        )
        # draw GTV-T contour
        dx = -ux * shift * sx
        dy = -uy * shift * sy
        sl = gtv_c[:, :, z_index]
        if sl.any():
            ax.contour(
                Xg + dx,
                Yg + dy,
                sl,
                levels=[0.5],
                colors=[color],
                linewidths=2,
            )
        # plot plane intersection line
        if abs(ny_f) > 1e-6:
            xs = np.array([x_min, x_max])
            ys = (C - nx_f * xs) / ny_f
        else:
            x_c = C / nx_f if abs(nx_f) > 1e-6 else 0
            xs = np.array([x_c, x_c])
            ys = np.array([y_min, y_max])
        ax.plot(xs, ys, "-", color=PLANE_COLOR, lw=2)
        # enforce equal axis limits and aspect
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # match extent reverse
        ax.set_aspect("equal")
        ax.axis("off")
        # ax.set_title(f'Shift={shift}' if shift else 'Central')
        # ax.set_title(f'Shift={shift}' if shift else 'Central')

    # save
    fig.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot_huber_loss(deltas, save_path, residual_range=(-5.0, 5.0), num_points=500):
    """Plot the Huber loss function for different delta values and save as a PDF.

    Parameters
    ----------
    deltas : array-like of float
        Sequence of delta thresholds for the Huber loss.
    save_path : str
        Filesystem path (including .pdf extension) where the figure will be saved.
    residual_range : tuple of float, optional
        (min, max) range of residual values over which to evaluate the loss.
        Default is (-5.0, 5.0).
    num_points : int, optional
        Number of points in the residual grid. Default is 500.

    The Huber loss Lδ(r) is defined as:
        Lδ(r) = { 0.5 * r²                    if |r| ≤ δ
               { δ * (|r| − 0.5 * δ)         if |r| > δ

    """
    # Create grid of residuals
    r = np.linspace(residual_range[0], residual_range[1], num_points)

    plt.figure()
    for δ in deltas:
        # Vectorised computation of Huber loss
        abs_r = np.abs(r)
        quadratic = 0.5 * r**2
        linear = δ * (abs_r - 0.5 * δ)
        huber = np.where(abs_r <= δ, quadratic, linear)

        plt.plot(r, huber, label=f"δ = {δ}")

    plt.title("Huber Loss for Various δ")
    plt.xlabel("Residual (r)")
    plt.ylabel("Huber Loss Lδ(r)")
    plt.legend()
    plt.tight_layout()

    # Save figure as PDF
    plt.savefig(save_path, format="pdf")
    plt.close()
