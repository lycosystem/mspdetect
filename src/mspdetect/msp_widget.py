"""Interactive Widget for Mid-Sagittal Plane Visualization"""

import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

import mspdetect as msp


def display_scrollable_views(
    struct_dict: dict,
    voxel_size,
    plane_coeffs_list=None,
    optimization_methods_list=None,
    shoulder_info=None,
    hide_msp=False,
):
    """Display interactive axial and coronal views of the image and structures, with optional plane overlays.

    Parameters
    ----------
    struct_dict : dict
        Dictionary of structures, must include 'Image'.
    voxel_size : tuple of float
        Spacing in mm along each axis.
    plane_coeffs_list : list of array-like, optional
        List of plane parameter arrays to overlay.
    optimization_methods_list : list of str, optional
        List of method names for legend.
    shoulder_info: dict
        Dict about the shoulder cropping in order to add some lines.

    """
    if plane_coeffs_list is None:
        plane_coeffs_list = []
    if optimization_methods_list is None:
        optimization_methods_list = []

    def _array(obj):
        return obj[0] if isinstance(obj, (tuple, list)) else obj

    image = _array(struct_dict["Image"])

    H, W, D = image.shape
    sy, sx, sz = voxel_size

    if shoulder_info is not None:
        shoulder_axis = shoulder_info.get("axis")
        shoulder_start = shoulder_info.get("start")
        shoulder_end = shoulder_info.get("end")
    else:
        shoulder_axis = None

    # Calculate center of mass using the Body structure for cropping
    # Use Body structure if available, otherwise fallback to geometric center
    if "Body" in struct_dict:
        try:
            body_mask = _array(struct_dict["Body"])
            if body_mask.any():
                # Get indices where Body structure is present
                coords = np.where(body_mask > 0)
                center_y = int(np.mean(coords[0]))
                center_x = int(np.mean(coords[1]))
                print(
                    f"Using Body structure center of mass: Y={center_y}, X={center_x}",
                )
            else:
                # Body structure exists but is empty
                center_y = H // 2
                center_x = W // 2
                print(
                    f"Body structure is empty, using geometric center: Y={center_y}, X={center_x}",
                )
        except Exception as e:
            # Error accessing Body structure
            center_y = H // 2
            center_x = W // 2
            print(
                f"Error accessing Body structure ({e}), using geometric center: Y={center_y}, X={center_x}",
            )
    else:
        # No Body structure available, use geometric center
        center_y = H // 2
        center_x = W // 2
        print(
            f"No Body structure found, using geometric center: Y={center_y}, X={center_x}",
        )

    # Define cropping around center of mass (150 voxels in each direction)
    crop_radius = 150

    # Calculate crop boundaries
    y_start = max(0, center_y - crop_radius)
    y_end = min(H, center_y + crop_radius)
    x_start = max(0, center_x - crop_radius)
    x_end = min(W, center_x + crop_radius)
    z_start = 0  # Keep full Z range
    z_end = D

    # Ensure we have at least the requested size if image is large enough
    if y_end - y_start < 2 * crop_radius and H >= 2 * crop_radius:
        if y_start == 0:
            y_end = min(H, 2 * crop_radius)
        elif y_end == H:
            y_start = max(0, H - 2 * crop_radius)

    if x_end - x_start < 2 * crop_radius and W >= 2 * crop_radius:
        if x_start == 0:
            x_end = min(W, 2 * crop_radius)
        elif x_end == W:
            x_start = max(0, W - 2 * crop_radius)

    # Ensure minimum dimensions for very small images
    min_size = 50
    if y_end - y_start < min_size:
        center_y = H // 2
        y_start = max(0, center_y - min_size // 2)
        y_end = min(H, y_start + min_size)
    if x_end - x_start < min_size:
        center_x = W // 2
        x_start = max(0, center_x - min_size // 2)
        x_end = min(W, x_start + min_size)

    print(f"Cropping around center of mass: Y=[{center_y}], X=[{center_x}]")
    print(
        f"Crop bounds: Y=[{y_start}:{y_end}], X=[{x_start}:{x_end}], Z=[{z_start}:{z_end}]",
    )
    print(
        f"Cropped size: {y_end - y_start} x {x_end - x_start} x {z_end - z_start} voxels",
    )

    # Crop the image
    image_cropped = image[y_start:y_end, x_start:x_end, z_start:z_end]
    H_crop, W_crop, D_crop = image_cropped.shape

    # physical axes for cropped image
    y_mm = np.arange(y_start, y_end) * sy
    x_mm = np.arange(x_start, x_end) * sx
    z_mm = np.arange(z_start, z_end) * sz  # Full Z range since not cropped

    # precompute meshgrids
    X_ax, Y_ax = np.meshgrid(x_mm, y_mm)  # for axial view
    X_cor, Z_cor = np.meshgrid(x_mm, z_mm)  # for coronal view

    # colors
    plane_colors = ["red", "purple", "cyan", "lime", "magenta"]
    struct_names = [s for s in struct_dict if s != "Image"]
    cmap10 = plt.cm.get_cmap("tab10").colors
    color_cycle = {n: cmap10[i] for i, n in enumerate(struct_names)}

    def view_slice_axial(z_index: int, y_line: int):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        # --- Axial ---
        ax0 = ax[0]
        ax0.imshow(
            image_cropped[:, :, z_index],
            cmap="gray",
            extent=[x_mm[0], x_mm[-1], y_mm[-1], y_mm[0]],
        )
        ax0.set_title(f"Axial z={z_index} (z={z_index * sz:.1f} mm)")
        ax0.set_xlabel("X (sagittal) mm")
        ax0.set_ylabel("Y (coronal) mm")
        # horizontal guide at Y = y_line
        y0 = y_mm[y_line]
        ax0.axhline(y=y0, color="yellow", linestyle="--")
        # structure contours
        for name in struct_names:
            if name != "Bones":
                mask = _array(struct_dict[name])
                # Crop the structure mask to match the image cropping
                mask_cropped = mask[y_start:y_end, x_start:x_end, z_start:z_end]
                sl = mask_cropped[:, :, z_index]
                if sl.any():
                    ax0.contour(
                        X_ax, Y_ax, sl, colors=[color_cycle[name]], linewidths=1
                    )
            if name == "Bones":
                mask = _array(struct_dict[name])
                # Crop the structure mask to match the image cropping
                mask_cropped = mask[y_start:y_end, x_start:x_end, z_start:z_end]
                sl = mask_cropped[:, :, z_index]
                if sl.any():
                    bone = sl > 0
                    bone_ma = np.ma.masked_where(~bone, bone.astype(float))
                    bone_cmap = ListedColormap([color_cycle["Bones"]])
                    bone_cmap.set_bad((0, 0, 0, 0))
                    ax0.imshow(
                        bone_ma,
                        cmap=bone_cmap,
                        extent=[x_mm[0], x_mm[-1], y_mm[-1], y_mm[0]],
                        alpha=0.8,
                        interpolation="nearest",
                        vmin=0,
                        vmax=1,
                    )
        if not hide_msp:
            # planes
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
                    colors=[plane_colors[i % len(plane_colors)]],
                    linewidths=1,
                )
        # legend
        handles = []
        for n in struct_names:
            handles.append(Line2D([0], [0], color=color_cycle[n], lw=2, label=n))
        if not hide_msp:
            for i, method in enumerate(optimization_methods_list):
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=plane_colors[i % len(plane_colors)],
                        lw=2,
                        label=method,
                    ),
                )

        if shoulder_axis is not None:
            if shoulder_axis == 0:  # Y-axis
                ax0.axhline(
                    y=shoulder_start * sy,
                    color="green",
                    linestyle="--",
                    label="Shoulder crop",
                )
                ax0.axhline(y=shoulder_end * sy, color="green", linestyle="--")
            elif shoulder_axis == 1:  # X-axis
                ax0.axvline(
                    x=shoulder_start * sx,
                    color="green",
                    linestyle="--",
                    label="Shoulder crop",
                )
                ax0.axvline(x=shoulder_end * sx, color="green", linestyle="--")

        ax0.legend(handles=handles, loc="upper right")

        # --- Coronal View ---
        ax1 = ax[1]

        # Extract coronal slice (using same cropping as axial view)
        sl_full = image_cropped[y_line, :, :].T  # Transpose for proper orientation

        # Display the coronal slice
        ax1.imshow(
            sl_full,
            cmap="gray",
            origin="lower",
            extent=[x_mm[0], x_mm[-1], z_mm[0], z_mm[-1]],
        )
        ax1.set_title(
            f"Coronal y={y_line + y_start} (y={(y_line + y_start) * sy:.1f} mm)",
        )
        ax1.set_xlabel("X (sagittal) mm")
        ax1.set_ylabel("Z (axial) mm")

        # Guide line at current Z slice
        z0 = z_mm[z_index]
        ax1.axhline(y=z0, color="yellow", linestyle="--")

        # Overlay structure contours (using same coordinate system as axial)
        for name in struct_names:
            if name != "Bones":
                mask = _array(struct_dict[name])
                mask_cropped = mask[y_start:y_end, x_start:x_end, z_start:z_end]
                slm_full = mask_cropped[
                    y_line, :, :
                ].T  # Transpose for proper orientation
                if slm_full.any():
                    ax1.contour(
                        X_cor,
                        Z_cor,
                        slm_full,
                        colors=[color_cycle[name]],
                        linewidths=1,
                    )

            if name == "Bones":
                mask = _array(struct_dict[name])
                mask_cropped = mask[y_start:y_end, x_start:x_end, z_start:z_end]
                slm_full = mask_cropped[
                    y_line, :, :
                ].T  # Transpose for proper orientation
                bone = slm_full > 0
                bone_ma = np.ma.masked_where(~bone, bone.astype(float))
                # One-color colormap for bones
                bone_cmap = ListedColormap([color_cycle["Bones"]])
                bone_cmap.set_bad((0, 0, 0, 0))  # transparent for masked pixels

                if slm_full.any():
                    # ax1.contourf(
                    #     X_cor,
                    #     Z_cor,
                    #     slm_full.astype(float),
                    #     levels=[1],
                    #     colors=[color_cycle[name]],
                    #     alpha=0.25,
                    # )
                    ax1.imshow(
                        bone_ma,
                        cmap=bone_cmap,
                        origin="lower",
                        extent=[x_mm[0], x_mm[-1], z_mm[0], z_mm[-1]],
                        alpha=0.8,
                        interpolation="nearest",
                        vmin=0,
                        vmax=1,
                    )

        # Overlay plane intersections (using same coordinate system as axial)
        if not hide_msp:
            y_phys = (y_line + y_start) * sy
            for i, (theta, phi, L) in enumerate(plane_coeffs_list):
                nx = np.sin(phi) * np.cos(theta)
                ny = np.sin(phi) * np.sin(theta)
                nz = np.cos(phi)
                F = nx * X_cor + ny * y_phys + nz * Z_cor - L
                ax1.contour(
                    X_cor,
                    Z_cor,
                    F,
                    levels=[0],
                    colors=[plane_colors[i % len(plane_colors)]],
                    linewidths=1,
                )

        if shoulder_axis == 1:  # X-axis
            ax1.axvline(
                x=shoulder_start * sx,
                color="green",
                linestyle="--",
                label="Shoulder crop",
            )
            ax1.axvline(x=shoulder_end * sx, color="green", linestyle="--")

        plt.show()

    # interactive sliders for Z (axial) and Y (coronal) - adjusted for cropped dimensions
    widgets.interact(
        view_slice_axial,
        z_index=widgets.IntSlider(
            min=0,
            max=D_crop - 1,
            step=1,
            value=D_crop // 2,
            description="Z slice",
            continuous_update=False,
        ),
        y_line=widgets.IntSlider(
            min=0,
            max=H_crop - 1,
            step=1,
            value=H_crop // 2,
            description="Y slice",
            continuous_update=False,
        ),
    )


def scrollable_patient(
    patient_path: str | None,
    config_path: str | None,
    result_path: str | None,
    crop_reference: str | None = "GTVp",
    crop_margin: int | None = 0,
    crop_mode: str | None = "margins",
    fixed_size: int | None = None,
    hide_msp: bool | None = False,
    show_bone: bool | None = False,
) -> None:
    """Display scrollable views of one patient"""
    config = msp.pipeline.load_config(config_path)
    pipeline_config = config["pipeline"]
    structure_name_mapping = pipeline_config.get("structure_name_mapping", None)

    # Load patient data
    struct_dict = msp.io.load_patient_structures(
        patient_path,
        pipeline_config["structure_names"],
        structure_name_mapping,
    )
    if show_bone:
        HU_range = pipeline_config.get("HU_range")
        voxel_size = struct_dict["Image"][1]
        struct_dict_copy = {}
        for key, item in struct_dict.items():
            struct_dict_copy[key] = item[0]
        proc_image, bone_ct = msp.preprocessing.preprocess_bone_image(
            struct_dict_copy, HU_range
        )
        struct_dict["Bones"] = bone_ct

    if not struct_dict:
        logger.warning(f"No matching files found in {patient_path}. Skipping...")
        return

    voxel_size = struct_dict["Image"][1]

    # Try to load full-image parameters
    full_param_path = os.path.join(result_path, "optimized_parameter_full_image.npy")

    if os.path.exists(full_param_path):
        params = np.load(full_param_path, mmap_mode="r").copy()

    else:
        # Use cropped params and convert
        cropped_param_path = os.path.join(result_path, "parameter_array.npy")
        params_cropped = np.load(cropped_param_path, mmap_mode="r")[-1].copy()
        _, _, crop_info = msp.preprocessing.crop_patient_volumes(
            struct_dict,
            pipeline_config["slice_axis"],
            pipeline_config["slice_range"],
            pipeline_config["margins"],
            pipeline_config["reference"],
            pipeline_config.get("crop_mode", None),
            pipeline_config.get("fixed_size", None),
            pipeline_config.get("exclude_mandible", False),
            pipeline_config.get("crop_shoulders", None),
        )
        params = msp.geometry.crop_to_full_coords(
            params_cropped,
            crop_info,
            voxel_size,
        )

    logger.info(f"loaded params {params}")

    struct_dict_cropped, _, crop_info_vis = msp.preprocessing.crop_patient_volumes(
        struct_dict,
        slice_axis=2,
        slice_range=None,
        margins=crop_margin,
        reference=crop_reference,
        crop_mode=crop_mode,
        fixed_size=fixed_size,
        exclude_mandible=pipeline_config.get("exclude_mandible", False),
    )

    # After loading params (around line 326)
    paramsfulltest = np.load(full_param_path, mmap_mode="r").copy()
    print(f"Loaded FULL params: {paramsfulltest}")

    # After crop (around line 362)
    print(f"crop_info_vis: {crop_info_vis}")
    print(f"struct_dict_cropped Image shape: {struct_dict_cropped['Image'].shape}")

    plane_params_cropped = msp.geometry.full_to_crop_coords(
        params,
        crop_info_vis,
        voxel_size,
    )
    # After conversion (around line 366)
    print(f"Converted to CROPPED params: {plane_params_cropped}")
    print(f"  - Plane offset L: {plane_params_cropped[2]:.2f} mm")

    crop_shoulder = pipeline_config.get("crop_shoulders")
    if crop_shoulder is not None:
        crop_info = msp.io.load_crop_info(result_path)
        shoulder_info = crop_info.get("shoulder_axis", None)
    else:
        shoulder_info = None

    display_scrollable_views(
        struct_dict_cropped,
        voxel_size,
        plane_coeffs_list=[plane_params_cropped],
        optimization_methods_list=None,
        shoulder_info=shoulder_info,
        hide_msp=hide_msp,
    )
    return


def adjustable_patient(
    patient_path: str | None,
    config_path: str | None,
    result_path: str | None,
    crop_reference: str | None = "GTVp",
    crop_margin: int | None = 0,
    crop_mode: str | None = "margins",
    fixed_size: int | None = None,
) -> None:
    """Display scrollable views of one patient"""
    config = msp.pipeline.load_config(config_path)
    pipeline_config = config["pipeline"]
    structure_name_mapping = pipeline_config.get("structure_name_mapping", None)

    # Load patient data
    struct_dict = msp.io.load_patient_structures(
        patient_path,
        pipeline_config["structure_names"],
        structure_name_mapping,
    )

    if not struct_dict:
        logger.warning(f"No matching files found in {patient_path}. Skipping...")
        return None

    voxel_size = struct_dict["Image"][1]

    # Try to load full-image parameters
    full_param_path = os.path.join(result_path, "optimized_parameter_full_image.npy")

    if os.path.exists(full_param_path):
        params = np.load(full_param_path, mmap_mode="r").copy()

    else:
        # Use cropped params and convert
        cropped_param_path = os.path.join(result_path, "parameter_array.npy")
        params_cropped = np.load(cropped_param_path, mmap_mode="r")[-1].copy()
        _, _, crop_info = msp.preprocessing.crop_patient_volumes(
            struct_dict,
            pipeline_config["slice_axis"],
            pipeline_config["slice_range"],
            pipeline_config["margins"],
            pipeline_config["reference"],
            pipeline_config.get("crop_mode", None),
            pipeline_config.get("fixed_size", None),
            pipeline_config.get("exclude_mandible", False),
            pipeline_config.get("crop_shoulders", None),
        )
        params = msp.geometry.crop_to_full_coords(
            params_cropped,
            crop_info,
            voxel_size,
        )

    logger.info(f"loaded params {params}")

    struct_dict_cropped, _, crop_info_vis = msp.preprocessing.crop_patient_volumes(
        struct_dict,
        slice_axis=2,
        slice_range=None,
        margins=crop_margin,
        reference=crop_reference,
        crop_mode=crop_mode,
        fixed_size=fixed_size,
        exclude_mandible=pipeline_config.get("exclude_mandible", False),
    )

    # After loading params (around line 326)
    paramsfulltest = np.load(full_param_path, mmap_mode="r").copy()
    print(f"Loaded FULL params: {paramsfulltest}")

    # After crop (around line 362)
    print(f"crop_info_vis: {crop_info_vis}")
    print(f"struct_dict_cropped Image shape: {struct_dict_cropped['Image'].shape}")

    plane_params_cropped = msp.geometry.full_to_crop_coords(
        params,
        crop_info_vis,
        voxel_size,
    )
    # After conversion (around line 366)
    print(f"Converted to CROPPED params: {plane_params_cropped}")
    print(f"  - Plane offset L: {plane_params_cropped[2]:.2f} mm")

    crop_shoulder = pipeline_config.get("crop_shoulders")
    if crop_shoulder is not None:
        crop_info = msp.io.load_crop_info(result_path)
        shoulder_info = crop_info.get("shoulder_axis", None)
    else:
        shoulder_info = None

    display_interactive_plane_widget(
        struct_dict_cropped,
        voxel_size,
        initial_plane_params=plane_params_cropped,
    )
    return crop_info_vis, voxel_size


def display_interactive_plane_widget(
    struct_dict: dict,
    voxel_size: tuple,
    initial_plane_params: tuple,
    theta_range: tuple = (-180, 180),
    phi_range: tuple = (0, 180),
    L_range: tuple = None,
):
    """Display interactive axial and coronal views with sliders to adjust plane parameters in real-time.

    Parameters
    ----------
    struct_dict : dict
        Dictionary of structures, must include 'Image'.
    voxel_size : tuple of float
        Spacing in mm along each axis (sy, sx, sz).
    initial_plane_params : tuple
        Initial plane parameters (theta_rad, phi_rad, L_mm).
    theta_range : tuple
        Range for theta slider in degrees (min, max). Default (-180, 180).
    phi_range : tuple
        Range for phi slider in degrees (min, max). Default (0, 180).
    L_range : tuple, optional
        Range for L slider in mm (min, max). If None, auto-computed from image bounds.

    """

    def _array(obj):
        return obj[0] if isinstance(obj, (tuple, list)) else obj

    # Load image and get dimensions
    image = _array(struct_dict["Image"])
    H, W, D = image.shape
    sy, sx, sz = voxel_size

    # Convert initial parameters to degrees for display
    theta_init_deg = np.rad2deg(initial_plane_params[0])
    phi_init_deg = np.rad2deg(initial_plane_params[1])
    L_init = initial_plane_params[2]

    # Auto-compute L range if not provided
    if L_range is None:
        # Estimate reasonable L range based on image physical dimensions
        max_dim = max(H * sy, W * sx, D * sz)
        L_range = (-max_dim, max_dim)

    # Calculate center of mass using the Body structure for cropping
    # Use Body structure if available, otherwise fallback to geometric center
    if "Body" in struct_dict:
        try:
            body_mask = _array(struct_dict["Body"])
            if body_mask.any():
                # Get indices where Body structure is present
                coords = np.where(body_mask > 0)
                center_y = int(np.mean(coords[0]))
                center_x = int(np.mean(coords[1]))
                print(
                    f"Using Body structure center of mass: Y={center_y}, X={center_x}",
                )
            else:
                # Body structure exists but is empty
                center_y = H // 2
                center_x = W // 2
                print(
                    f"Body structure is empty, using geometric center: Y={center_y}, X={center_x}",
                )
        except Exception as e:
            # Error accessing Body structure
            center_y = H // 2
            center_x = W // 2
            print(
                f"Error accessing Body structure ({e}), using geometric center: Y={center_y}, X={center_x}",
            )
    else:
        # No Body structure available, use geometric center
        center_y = H // 2
        center_x = W // 2
        print(
            f"No Body structure found, using geometric center: Y={center_y}, X={center_x}",
        )

    # Define cropping around center of mass (150 voxels in each direction)
    crop_radius = 150

    # Calculate crop boundaries
    y_start = max(0, center_y - crop_radius)
    y_end = min(H, center_y + crop_radius)
    x_start = max(0, center_x - crop_radius)
    x_end = min(W, center_x + crop_radius)
    z_start = 0  # Keep full Z range
    z_end = D

    # Ensure we have at least the requested size if image is large enough
    if y_end - y_start < 2 * crop_radius and H >= 2 * crop_radius:
        if y_start == 0:
            y_end = min(H, 2 * crop_radius)
        elif y_end == H:
            y_start = max(0, H - 2 * crop_radius)

    if x_end - x_start < 2 * crop_radius and W >= 2 * crop_radius:
        if x_start == 0:
            x_end = min(W, 2 * crop_radius)
        elif x_end == W:
            x_start = max(0, W - 2 * crop_radius)

    # Ensure minimum dimensions for very small images
    min_size = 50
    if y_end - y_start < min_size:
        center_y = H // 2
        y_start = max(0, center_y - min_size // 2)
        y_end = min(H, y_start + min_size)
    if x_end - x_start < min_size:
        center_x = W // 2
        x_start = max(0, center_x - min_size // 2)
        x_end = min(W, x_start + min_size)

    print(f"Cropping around center of mass: Y=[{center_y}], X=[{center_x}]")
    print(
        f"Crop bounds: Y=[{y_start}:{y_end}], X=[{x_start}:{x_end}], Z=[{z_start}:{z_end}]",
    )
    print(
        f"Cropped size: {y_end - y_start} x {x_end - x_start} x {z_end - z_start} voxels",
    )

    # Crop the image
    image_cropped = image[y_start:y_end, x_start:x_end, z_start:z_end]
    H_crop, W_crop, D_crop = image_cropped.shape

    # Physical coordinates for cropped image
    y_mm = np.arange(y_start, y_end) * sy
    x_mm = np.arange(x_start, x_end) * sx
    z_mm = np.arange(z_start, z_end) * sz

    # Meshgrids for plotting
    X_ax, Y_ax = np.meshgrid(x_mm, y_mm)  # for axial view
    X_cor, Z_cor = np.meshgrid(x_mm, z_mm)  # for coronal view

    # Colors for structures
    struct_names = [s for s in struct_dict if s != "Image"]
    cmap10 = plt.colormaps["tab10"].colors
    color_cycle = {n: cmap10[i % len(cmap10)] for i, n in enumerate(struct_names)}

    def update_plot(theta_deg, phi_deg, L_mm, z_index, y_line):
        """Update the plot with new plane parameters"""
        # Convert degrees to radians
        theta_rad = np.deg2rad(theta_deg)
        phi_rad = np.deg2rad(phi_deg)

        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)

        # --- AXIAL VIEW ---
        ax0 = ax[0]

        # Check bounds for z_index
        if z_index >= D_crop:
            z_index = D_crop // 2

        ax0.imshow(
            image_cropped[:, :, z_index],
            cmap="gray",
            extent=[x_mm[0], x_mm[-1], y_mm[-1], y_mm[0]],
        )
        ax0.set_title(
            f"Axial z={z_index} (z={z_index * sz:.1f} mm)  "
            + f"θ={theta_deg:.1f}°, φ={phi_deg:.1f}°, L={L_mm:.1f}mm",
        )
        ax0.set_xlabel("X (sagittal) mm")
        ax0.set_ylabel("Y (coronal) mm")

        # Horizontal guide line at current Y slice
        if y_line < len(y_mm):
            y0 = y_mm[y_line]
            ax0.axhline(y=y0, color="yellow", linestyle="--", alpha=0.7)

        # Structure contours on axial view
        for name in struct_names:
            mask = _array(struct_dict[name])
            mask_cropped = mask[y_start:y_end, x_start:x_end, z_start:z_end]
            if z_index < mask_cropped.shape[2]:
                sl = mask_cropped[:, :, z_index]
                if sl.any():
                    ax0.contour(
                        X_ax,
                        Y_ax,
                        sl,
                        colors=[color_cycle[name]],
                        linewidths=1.5,
                        alpha=0.8,
                    )

        # Plane intersection on axial view
        nx = np.sin(phi_rad) * np.cos(theta_rad)
        ny = np.sin(phi_rad) * np.sin(theta_rad)
        nz = np.cos(phi_rad)
        C = L_mm - nz * (z_index * sz)
        F_axial = nx * X_ax + ny * Y_ax - C
        ax0.contour(X_ax, Y_ax, F_axial, levels=[0], colors=["red"], linewidths=2)

        # --- CORONAL VIEW ---
        ax1 = ax[1]

        # Check bounds for y_line
        if y_line >= H_crop:
            y_line = H_crop // 2

        # Extract coronal slice (using same cropping as axial view)
        sl_full = image_cropped[y_line, :, :].T  # Transpose for proper orientation

        ax1.imshow(
            sl_full,
            cmap="gray",
            origin="lower",
            extent=[x_mm[0], x_mm[-1], z_mm[0], z_mm[-1]],
        )
        ax1.set_title(
            f"Coronal y={y_line + y_start} (y={(y_line + y_start) * sy:.1f} mm)",
        )
        ax1.set_xlabel("X (sagittal) mm")
        ax1.set_ylabel("Z (axial) mm")

        # Vertical guide line at current Z slice
        z0 = z_mm[z_index] if z_index < len(z_mm) else z_mm[len(z_mm) // 2]
        ax1.axhline(y=z0, color="yellow", linestyle="--", alpha=0.7)

        # Structure contours on coronal view
        for name in struct_names:
            mask = _array(struct_dict[name])
            mask_cropped = mask[y_start:y_end, x_start:x_end, z_start:z_end]
            if y_line < mask_cropped.shape[0]:
                slm_full = mask_cropped[
                    y_line,
                    :,
                    :,
                ].T  # Transpose for proper orientation
                if slm_full.any():
                    ax1.contour(
                        X_cor,
                        Z_cor,
                        slm_full,
                        colors=[color_cycle[name]],
                        linewidths=1.5,
                        alpha=0.8,
                    )

        # Plane intersection on coronal view
        y_phys = (y_line + y_start) * sy
        F_coronal = nx * X_cor + ny * y_phys + nz * Z_cor - L_mm
        ax1.contour(X_cor, Z_cor, F_coronal, levels=[0], colors=["red"], linewidths=2)

        # Add legend
        handles = []
        for name in struct_names:
            handles.append(
                plt.Line2D([0], [0], color=color_cycle[name], lw=2, label=name),
            )
        handles.append(plt.Line2D([0], [0], color="red", lw=2, label="MSP"))
        ax0.legend(handles=handles, loc="upper right", fontsize=10)

        plt.show()

    # Create interactive widget with sliders
    theta_slider = widgets.FloatSlider(
        value=theta_init_deg,
        min=theta_range[0],
        max=theta_range[1],
        step=0.1,
        description="θ (deg):",
        continuous_update=False,
        style={"description_width": "initial"},
    )

    phi_slider = widgets.FloatSlider(
        value=phi_init_deg,
        min=phi_range[0],
        max=phi_range[1],
        step=0.1,
        description="φ (deg):",
        continuous_update=False,
        style={"description_width": "initial"},
    )

    L_slider = widgets.FloatSlider(
        value=L_init,
        min=L_range[0],
        max=L_range[1],
        step=1.0,
        description="L (mm):",
        continuous_update=False,
        style={"description_width": "initial"},
    )

    z_slider = widgets.IntSlider(
        value=D_crop // 2,
        min=0,
        max=D_crop - 1,
        step=1,
        description="Z slice:",
        continuous_update=False,
        style={"description_width": "initial"},
    )

    y_slider = widgets.IntSlider(
        value=H_crop // 2,
        min=0,
        max=H_crop - 1,
        step=1,
        description="Y slice:",
        continuous_update=False,
        style={"description_width": "initial"},
    )

    # Create interactive widget
    widgets.interact(
        update_plot,
        theta_deg=theta_slider,
        phi_deg=phi_slider,
        L_mm=L_slider,
        z_index=z_slider,
        y_line=y_slider,
    )

    # Display current parameters
    def get_current_params():
        """Get current plane parameters in radians"""
        return (
            np.deg2rad(theta_slider.value),
            np.deg2rad(phi_slider.value),
            L_slider.value,
        )

    # Store reference to sliders for external access
    display_interactive_plane_widget.sliders = {
        "theta": theta_slider,
        "phi": phi_slider,
        "L": L_slider,
        "z": z_slider,
        "y": y_slider,
    }
    display_interactive_plane_widget.get_current_params = get_current_params
    crop_info = {
        "x_axis": {"start": x_start, "end": x_end, "axis": 1},
        "y_axis": {"start": y_start, "end": y_end, "axis": 0},
    }
    return


# Simple function to get current widget parameters - call this directly!
def get_params():
    """Simple function to get current widget parameters.
    Just run: get_params()
    """
    try:
        if hasattr(display_interactive_plane_widget, "sliders"):
            sliders = display_interactive_plane_widget.sliders

            # Get values
            theta_deg = sliders["theta"].value
            phi_deg = sliders["phi"].value
            L_mm = sliders["L"].value
            z_idx = sliders["z"].value
            y_idx = sliders["y"].value

            # Convert to radians
            theta_rad = np.deg2rad(theta_deg)
            phi_rad = np.deg2rad(phi_deg)

            # Print formatted output
            print("=" * 60)
            print("CURRENT WIDGET PARAMETERS")
            print("=" * 60)
            print(f"θ (Theta): {theta_deg:7.2f}° = {theta_rad:8.4f} rad")
            print(f"φ (Phi):   {phi_deg:7.2f}° = {phi_rad:8.4f} rad")
            print(f"L (Dist):  {L_mm:7.2f} mm")
            print(f"Z slice:   {z_idx}")
            print(f"Y line:    {y_idx}")
            print("-" * 60)
            print("COPY-READY FORMATS:")
            print("-" * 60)
            param_tuple_rad = f"({theta_rad:.4f}, {phi_rad:.4f}, {L_mm:.2f})"
            param_tuple_deg = f"({theta_deg:.2f}, {phi_deg:.2f}, {L_mm:.2f})"
            print(f"Radians: {param_tuple_rad}")
            print(f"Degrees: {param_tuple_deg}")
            print("=" * 60)

            return {
                "theta_rad": theta_rad,
                "phi_rad": phi_rad,
                "L_mm": L_mm,
                "theta_deg": theta_deg,
                "phi_deg": phi_deg,
                "z_index": z_idx,
                "y_line": y_idx,
                "tuple_rad": (theta_rad, phi_rad, L_mm),
                "tuple_deg": (theta_deg, phi_deg, L_mm),
            }
        print("❌ Widget not found!")
        print("Make sure you have:")
        print("1. Executed the display_interactive_plane_widget function")
        print("2. The widget is currently active")
        return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None
