"""This module defines possible optimization functions, inizialises and performs optimisation of the MSP."""

import multiprocessing
import os
import time

import numpy as np
import scipy as sp

from mspdetect import geometry, visualization


def compute_signed_distances(params_array, image, voxel_size):
    """Compute signed distances from all nonzero voxels in the image to a plane.

    Parameters
    ----------
    params_array : array-like of shape (3,)
        Plane parameters [azimuthal, polar, L].
    image : np.ndarray
        Binary or intensity image.
    voxel_size : array-like of shape (3,)
        Physical voxel size for each axis.

    Returns
    -------
    tuple
        d : np.ndarray
            Signed distances from each voxel to the plane.
        n : np.ndarray
            Normal vector of the plane.
        indices_coord_syst_phy : np.ndarray
            Physical coordinates of the voxels in the coordinate system.
        indices_image : np.ndarray
            Indices of nonzero voxels in the image.

    """
    azimuthal, polar, L = params_array

    indices_image = np.array(np.nonzero(image)).T

    indices_image_phy = indices_image * voxel_size

    indices_coord_syst_phy = np.stack(
        [
            indices_image_phy[:, 1],
            indices_image_phy[:, 0],
            indices_image_phy[:, 2],
        ],
        axis=1,
    )

    n = geometry.generate_normal(azimuthal, polar)

    d = indices_coord_syst_phy.dot(n) - L

    return d, n, indices_coord_syst_phy, indices_image


def quadratic_loss_function(diff):
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


def huber_loss_function(diff, delta=300):
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


def p_huber_loss(r, c=300.0, p=1.4):
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


def welsch_loss(r, c=200):
    """ρc(r) for Welsch."""
    factor = (r / c) ** 2
    return 0.5 * c**2 * (1.0 - np.exp(-factor))


def piecewise_loss(r):
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


def piecewise_intensity_loss(I_orig, I_m, threshold=2800):
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


def compute_delta(diff, dim, tol=1e-6, max_iter=100):
    """Solve for δ via the truncated-moment equation:
      (1/(n δ^2)) * sum(min(r_i^2, δ^2)) = (dim + z) / n,
    where z = log(n), using bisection.
    """
    n = diff.size
    r2 = diff**2
    z = np.log(n)

    def phi(delta):
        return (np.minimum(r2, delta**2).sum() / (n * delta**2)) - ((dim + z) / n)

    # Bounds
    low = np.finfo(float).eps
    high = np.sqrt(r2.sum() / (dim + z)) + tol

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if phi(mid) > 0:
            low = mid
        else:
            high = mid
        if (high - low) < tol * mid:
            break

    return 0.5 * (low + high)


def compute_objective(params_array, bone, interpolator_intensity, voxel_size, delta):
    """Compute the objective function value for plane symmetry optimization.

    Parameters
    ----------
    params_array : array-like of shape (3,)
        Plane parameters [azimuthal, polar, L].
    bone : np.ndarray
        Bone mask or intensity image.
    interpolator_intensity : RegularGridInterpolator
        Interpolator for the original image.
    voxel_size : array-like of shape (3,)
        Physical voxel size for each axis.
    delta : float
        Huber loss delta parameter.

    Returns
    -------
    float
        The mean Huber loss for the current plane parameters.

    """
    d, n, indices_coord_syst_phy, indices_image = compute_signed_distances(
        params_array,
        bone,
        voxel_size,
    )
    x_m_coord_syst_phy = indices_coord_syst_phy - 2 * d[:, None] * n[None, :]
    x_m_image_phy = np.array(
        [x_m_coord_syst_phy[:, 1], x_m_coord_syst_phy[:, 0], x_m_coord_syst_phy[:, 2]],
    ).T
    I_m = interpolator_intensity(x_m_image_phy)
    I_orig = bone[indices_image[:, 0], indices_image[:, 1], indices_image[:, 2]]
    diff = I_orig - I_m

    # f = np.sum(piecewise_intensity_loss(I_orig, I_m)).mean()

    # f= np.sum(quadratic_loss_function(diff)).mean()

    # # 5) estimate δ via truncated-moment
    # dim = params_array.size
    # delta_adapt = compute_delta(diff, dim)

    f = np.sum(huber_loss_function(diff, delta=delta)).mean()

    # f = np.sum(welsch_loss(diff, c=delta)).mean()  # Using Welsch loss instead of Huber
    # f = np.sum(p_huber_loss(diff, c=delta, p=1)).mean()  # Using generalized Huber loss
    # f = np.sum(piecewise_loss(diff)).mean()  # Using custom piecewise loss
    return f


def compute_objective_initialization(
    params_array,
    bone,
    interpolator_intensity,
    voxel_size,
):
    """Compute the objective function value for plane symmetry optimization.

    Parameters
    ----------
    params_array : array-like of shape (3,)
        Plane parameters [azimuthal, polar, L].
    bone : np.ndarray
        Bone mask or intensity image.
    interpolator_intensity : RegularGridInterpolator
        Interpolator for the original image.
    voxel_size : array-like of shape (3,)
        Physical voxel size for each axis.
    delta : float
        Huber loss delta parameter.

    Returns
    -------
    float
        The mean Huber loss for the current plane parameters.

    """
    d, n, indices_coord_syst_phy, indices_image = compute_signed_distances(
        params_array,
        bone,
        voxel_size,
    )
    x_m_coord_syst_phy = indices_coord_syst_phy - 2 * d[:, None] * n[None, :]
    x_m_image_phy = np.array(
        [x_m_coord_syst_phy[:, 1], x_m_coord_syst_phy[:, 0], x_m_coord_syst_phy[:, 2]],
    ).T
    I_m = interpolator_intensity(x_m_image_phy)
    I_orig = bone[indices_image[:, 0], indices_image[:, 1], indices_image[:, 2]]
    diff = I_orig - I_m

    diff_shifted = diff - np.mean(diff)
    mad = np.median(np.abs(diff_shifted))
    sigma_hat = 1.4826 * mad
    delta = 1.345 * sigma_hat

    # visualization.plot_shifted_residuals_hist(diff, delta)

    f = np.sum(quadratic_loss_function(diff)).mean()

    return f, delta


def parameter_initialization(
    image,
    bone,
    output_path,
    interpolator_intensity,
    voxel_size,
    azimuthal_deg_range=(0, 90),
    polar_deg_range=(90, 10),
    initialization_steps=10,
):
    """Perform grid search initialization for plane parameters by evaluating the objective
    function over a range of azimuthal and polar angles.

    Returns the best plane parameters [azimuthal, polar, L].
    """
    start = time.time()

    com_vox = sp.ndimage.center_of_mass(bone)  # in voxel indices (z, y, x)
    com_phy = np.array([com_vox[1], com_vox[0], com_vox[2]]) * voxel_size

    # 2) Build angular search ranges (in radians)
    az_cent_deg, az_half_deg = azimuthal_deg_range
    pol_cent_deg, pol_half_deg = polar_deg_range
    az_cent_rad, az_half_rad = np.deg2rad([az_cent_deg, az_half_deg])
    pol_cent_rad, pol_half_rad = np.deg2rad([pol_cent_deg, pol_half_deg])

    azimuthal_angles = np.linspace(
        az_cent_rad - az_half_rad,
        az_cent_rad + az_half_rad,
        initialization_steps,
    )
    polar_angles = np.linspace(
        pol_cent_rad - pol_half_rad,
        pol_cent_rad + pol_half_rad,
        initialization_steps,
    )

    os.makedirs(output_path, exist_ok=True)
    mse_file = os.path.join(output_path, "Initialization_obj_fun.npy")
    params_file = os.path.join(output_path, "Initialization_plane_params.npy")
    delta_file = os.path.join(output_path, "Initialization_delta.npy")

    # 3) If not already computed, run grid search
    if not (os.path.exists(mse_file) and os.path.exists(params_file)):
        print("Starting parameter initialization...")

        # Create grid of angles
        theta_grid, phi_grid = np.meshgrid(
            azimuthal_angles,
            polar_angles,
            indexing="ij",
        )  # both shape (N, N)

        # Compute normals for each (θ, φ)
        n = geometry.generate_normal(theta_grid, phi_grid)  # shape (N, N, 3)

        # Compute offsets L_{ij} = n_{ij} · com_phy
        L_grid = np.tensordot(n, com_phy, axes=([-1], [0]))  # shape (N, N)

        # Flatten into list of plane parameters
        planes = np.stack(
            [
                theta_grid.ravel(),
                phi_grid.ravel(),
                L_grid.ravel(),
            ],
            axis=1,
        )  # shape (N*N, 3)

        # Evaluate objective for each plane: returns list of (f, δ)
        results = [
            compute_objective_initialization(
                p,
                bone,
                interpolator_intensity,
                voxel_size,
            )
            for p in planes
        ]
        mse_vals, delta_vals = zip(*results, strict=False)
        mse_array = np.array(mse_vals)  # shape (N*N,)
        delta_array = np.array(delta_vals)  # shape (N*N,)

        # Find best index
        best_idx = np.argmin(mse_array)
        best_f = mse_array[best_idx]
        best_delta = delta_array[best_idx]
        print(f"Minimum objective f = {best_f:.4f} at index {best_idx}")
        print(f"Corresponding δ = {best_delta:.4f}")

        # Save for reuse
        np.save(mse_file, mse_array)
        np.save(params_file, planes)
        np.save(delta_file, delta_array)

        end = time.time()
        print(f"Time taken for initialization: {end - start:.2f} s")

    else:
        mse_array = np.load(mse_file)  # shape (N*N,)
        planes = np.load(params_file)  # shape (N*N, 3)
        delta_array = np.load(delta_file)  # shape (N*N,)
        N = initialization_steps
        theta_grid = np.linspace(
            az_cent_rad - az_half_rad,
            az_cent_rad + az_half_rad,
            N,
        )
        phi_grid = np.linspace(
            pol_cent_rad - pol_half_rad,
            pol_cent_rad + pol_half_rad,
            N,
        )

    # 4) Select best parameters from saved results
    best_idx = np.argmin(mse_array)
    best_params = planes[best_idx]  # [azimuthal, polar, L]
    best_delta = delta_array[best_idx]

    # 5) Optional: plot heatmap of MSE over (θ, φ)
    mse_grid = mse_array.reshape(initialization_steps, initialization_steps)
    visualization.plot_mse_heatmap(
        azimuthal_angles,
        polar_angles,
        mse_grid,
        title=(
            f"MSE Heatmap (θ, φ) – optimum at:\n"
            f"θ={np.rad2deg(best_params[0]):.2f}°, "
            f"φ={np.rad2deg(best_params[1]):.2f}°, "
            f"L={best_params[2]:.2f}\n"
            f"MSE={mse_grid.flat[best_idx]:.2f}"
        ),
        output_path=output_path,
    )
    print(
        f"Best estimate of initial plane parameters: θ = {np.rad2deg(best_params[0]):.2f}°, φ = {np.rad2deg(best_params[1]):.2f}°, L = {best_params[2]:.2f}",
    )

    # z_slice = int(round(com_vox[0]))  # axial index through the COM

    # # Convert planes (shape (N²,3)) to a list of tuples
    # all_planes = [tuple(p) for p in planes]

    # plot_and_save_axial_slice(
    #     struct_dict={'Image': bone},sss
    #     voxel_size=voxel_size,
    #     z_indices=6,
    #     save_path=os.path.join(output_path, "all_sampled_planes.pdf"),
    #     crop_size=200,
    #     plane_coeffs_list=all_planes,
    #     optimization_methods_list=None,   # omit legend if too many planes
    #     figsize_per=(6,6)
    # )

    return best_params, best_delta


def optimize_plane(
    initial_params_array,
    image,
    interpolator_intensity,
    voxel_size,
    delta,
):
    """Optimize the plane parameters to minimize the objective function using BFGS.

    Parameters
    ----------
    initial_params_array : array-like of shape (3,)
        Initial guess for plane parameters [azimuthal, polar, L].
    image : np.ndarray
        The image or mask to optimize over.
    interpolator_intensity : RegularGridInterpolator
        Interpolator for the original image.
    voxel_size : array-like of shape (3,)
        Physical voxel size for each axis.
    delta : float
        Huber loss delta parameter.

    Returns
    -------
    OptimizeResult
        Result object from scipy.optimize.minimize, with additional attributes:
        - objective_value_list: list of objective values per iteration.
        - params_list: list of parameter arrays per iteration.

    """
    objective_value_list = []
    params_list = []

    def callback(xk):
        f_val = compute_objective(xk, image, interpolator_intensity, voxel_size, delta)
        objective_value_list.append(f_val)
        params_list.append(xk.copy())

    res = sp.optimize.minimize(
        compute_objective,
        x0=initial_params_array,
        args=(image, interpolator_intensity, voxel_size, delta),
        method="BFGS",
        jac=None,
        callback=callback,
    )
    params_list.append(res.x.copy())
    objective_value_list.append(res.fun)

    res.objective_value_list = objective_value_list
    res.params_list = params_list

    return res


def _optimize_single_initialization(args):
    """Worker function for multiprocessing optimization of a single initialization point.

    Parameters
    ----------
    args : tuple
        (index, plane_params, bone, interpolator_intensity, voxel_size,
         optimal_plane_params, theta_range, phi_range, L_range)

    Returns
    -------
    dict
        Optimization result dictionary

    """
    (
        index,
        p,
        bone,
        interpolator_intensity,
        voxel_size,
        optimal_plane_params,
        theta_range,
        phi_range,
        L_range,
    ) = args

    try:
        # Compute initial objective and delta
        f, delta = compute_objective_initialization(
            p,
            bone,
            interpolator_intensity,
            voxel_size,
        )

        # Run optimization from this initialization
        res = optimize_plane(p, bone, interpolator_intensity, voxel_size, delta)

        optimized_parameters = res.x
        optimized_objective_value = res.fun

        # Compute normalized error metric relative to optimal parameters
        E = np.sqrt(
            (optimized_parameters[0] - optimal_plane_params[0]) ** 2 / theta_range**2
            + (optimized_parameters[1] - optimal_plane_params[1]) ** 2 / phi_range**2
            + (optimized_parameters[2] - optimal_plane_params[2]) ** 2 / L_range**2,
        )

        return {
            "index": index,
            "initial_params": p.copy(),
            "optimized_params": optimized_parameters.copy(),
            "initial_objective": f,
            "optimized_objective": optimized_objective_value,
            "delta": delta,
            "success": res.success,
            "nit": res.nit,
            "error_metric": E,
        }

    except Exception as e:
        return {
            "index": index,
            "initial_params": p.copy(),
            "optimized_params": None,
            "initial_objective": f if "f" in locals() else np.inf,
            "optimized_objective": np.inf,
            "delta": delta if "delta" in locals() else np.inf,
            "success": False,
            "nit": 0,
            "error_metric": np.inf,
            "error": str(e),
        }


def robustness_test(
    image,
    bone,
    output_path,
    interpolator_intensity,
    voxel_size,
    azimuthal_deg_range=(0, 90),
    polar_deg_range=(90, 10),
    initialization_steps=10,
    optimal_plane_params=None,
    save_results=True,
    force_regenerate=False,
):
    """Perform grid search initialization for plane parameters by evaluating the objective
    function over a range of azimuthal and polar angles. Test robustness by comparing
    optimized results to known optimal parameters.

    Automatically checks for existing results and loads them if available, unless
    force_regenerate=True.

    Parameters
    ----------
    image : np.ndarray
        Input image data
    bone : np.ndarray
        Bone mask or segmentation
    output_path : str
        Directory to save results
    interpolator_intensity : RegularGridInterpolator
        Interpolator for the original image
    voxel_size : array-like
        Physical voxel size for each axis
    azimuthal_deg_range : tuple
        (center, half_range) for azimuthal angles in degrees
    polar_deg_range : tuple
        (center, half_range) for polar angles in degrees
    initialization_steps : int
        Number of steps in each angular direction
    optimal_plane_params : array-like, optional
        Known optimal plane parameters [azimuthal, polar, L] for comparison
    save_results : bool, optional
        Whether to save results to files for later analysis
    force_regenerate : bool, optional
        If True, regenerate results even if existing files are found

    Returns
    -------
    dict
        Dictionary containing optimization results, error metrics, and grid parameters

    """
    start_time = time.time()

    if optimal_plane_params is None:
        raise ValueError("optimal_plane_params must be provided for robustness testing")

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Define file paths for saved results
    results_file = os.path.join(output_path, "robustness_test_results.npz")
    metadata_file = os.path.join(output_path, "robustness_test_metadata.json")

    # Check if results already exist and match current parameters
    results_exist = os.path.exists(results_file) and os.path.exists(metadata_file)

    if results_exist and not force_regenerate:
        try:
            print("Checking for existing robustness test results...")

            # Load existing metadata to check if parameters match
            import json

            with open(metadata_file) as f:
                existing_metadata = json.load(f)

            # Check if parameters match (handle tuple/list conversion from JSON)
            current_params = {
                "azimuthal_deg_range": azimuthal_deg_range,
                "polar_deg_range": polar_deg_range,
                "initialization_steps": initialization_steps,
            }

            existing_params = existing_metadata.get("parameters", {})

            # Convert both to lists for comparison (JSON converts tuples to lists)
            def normalize_param(param):
                if isinstance(param, (tuple, list)):
                    return list(param)
                return param

            params_match = (
                normalize_param(existing_params.get("azimuthal_deg_range"))
                == normalize_param(current_params["azimuthal_deg_range"])
                and normalize_param(existing_params.get("polar_deg_range"))
                == normalize_param(current_params["polar_deg_range"])
                and existing_params.get("initialization_steps")
                == current_params["initialization_steps"]
            )

            if params_match:
                print("✓ Found existing results with matching parameters!")
                print(
                    f"  - Results from: {existing_metadata.get('timestamp', 'unknown time')}",
                )
                print(
                    f"  - Grid shape: {existing_metadata.get('grid_shape', 'unknown')}",
                )
                print(
                    f"  - Total optimizations: {existing_metadata.get('total_optimizations', 'unknown')}",
                )

                # Load existing results
                data = np.load(results_file)

                # Check if optimal_plane_params match (allowing for small numerical differences)
                existing_optimal = data["optimal_plane_params"]
                if np.allclose(existing_optimal, optimal_plane_params, atol=1e-6):
                    print("✓ Optimal plane parameters match existing results")

                    # Reconstruct results dictionary
                    optimization_results = []
                    for i in range(len(data["initial_params"])):
                        opt_params = data["optimized_params"][i]
                        if np.all(np.isnan(opt_params)):
                            opt_params = None
                        else:
                            opt_params = opt_params.copy()

                        optimization_results.append(
                            {
                                "initial_params": data["initial_params"][i].copy(),
                                "optimized_params": opt_params,
                                "initial_objective": data["initial_objectives"][i],
                                "optimized_objective": data["optimized_objectives"][i],
                                "delta": data["deltas"][i],
                                "success": bool(data["success_flags"][i]),
                                "nit": int(data["nit_counts"][i]),
                                "error_metric": data["error_metrics"][i],
                            },
                        )

                    # Compute ranges for compatibility
                    az_cent_deg, az_half_deg = azimuthal_deg_range
                    pol_cent_deg, pol_half_deg = polar_deg_range
                    az_half_rad = np.deg2rad(az_half_deg)
                    pol_half_rad = np.deg2rad(pol_half_deg)
                    theta_range = 2 * az_half_rad
                    phi_range = 2 * pol_half_rad
                    L_range = np.max(data["L_grid"]) - np.min(data["L_grid"])

                    results = {
                        "optimization_results": optimization_results,
                        "error_metrics": data["error_metrics"],
                        "azimuthal_angles": data["azimuthal_angles"],
                        "polar_angles": data["polar_angles"],
                        "theta_grid": data["theta_grid"],
                        "phi_grid": data["phi_grid"],
                        "L_grid": data["L_grid"],
                        "optimal_plane_params": data["optimal_plane_params"],
                        "statistics": existing_metadata["statistics"],
                        "ranges": {
                            "theta_range": theta_range,
                            "phi_range": phi_range,
                            "L_range": L_range,
                        },
                        "parameters": existing_metadata["parameters"],
                    }

                    print("✓ Successfully loaded existing results!")
                    print(
                        f"  - Success rate: {results['statistics']['success_rate']:.1%}",
                    )
                    print(f"  - Mean error: {results['statistics']['mean_error']:.4f}")

                    # Compute and print zero error percentage (compute if not available in saved results)
                    if "zero_error_percentage" in results["statistics"]:
                        zero_count = results["statistics"].get("zero_error_count", 0)
                        zero_pct = results["statistics"]["zero_error_percentage"]
                        total_samples = existing_metadata.get(
                            "total_optimizations",
                            "unknown",
                        )
                        print(
                            f"  - Near-zero error samples: {zero_count}/{total_samples} ({zero_pct:.1f}%)",
                        )
                    else:
                        # Compute zero error statistics from loaded error metrics
                        error_metrics = data["error_metrics"]
                        finite_errors = error_metrics[np.isfinite(error_metrics)]
                        zero_error_tolerance = 1e-1
                        zero_error_count = np.sum(finite_errors < zero_error_tolerance)
                        zero_error_percentage = (
                            zero_error_count / len(error_metrics)
                        ) * 100
                        total_samples = len(error_metrics)

                        # Update the statistics in the results
                        results["statistics"]["zero_error_count"] = zero_error_count
                        results["statistics"]["zero_error_percentage"] = (
                            zero_error_percentage
                        )

                        print(
                            f"  - Near-zero error samples: {zero_error_count}/{total_samples} ({zero_error_percentage:.1f}%) [computed from loaded data]",
                        )

                    # Generate heatmap from loaded results
                    print("Generating heatmap from existing results...")
                    visualization.plot_robustness_heatmap(
                        azimuthal_angles=results["azimuthal_angles"],
                        polar_angles=results["polar_angles"],
                        error_array=results["error_metrics"],
                        title="Robustness Test - Parameter Error vs Initialization (Loaded)",
                        output_path=output_path,
                    )

                    return results

                print("⚠ Optimal plane parameters don't match existing results")
                print("  Regenerating with new optimal parameters...")

            else:
                print("⚠ Parameters don't match existing results")
                print(f"  Current: {current_params}")
                print(f"  Existing: {existing_params}")
                print("  Regenerating with new parameters...")

        except Exception as e:
            print(f"⚠ Error loading existing results: {e}")
            print("  Regenerating results...")

    elif force_regenerate:
        print("🔄 Force regeneration requested - computing new results...")
    else:
        print("📊 No existing results found - computing new results...")

    # Generate new results (original computation logic)
    com_vox = sp.ndimage.center_of_mass(bone)  # in voxel indices (z, y, x)
    com_phy = np.array([com_vox[1], com_vox[0], com_vox[2]]) * voxel_size

    # Build angular search ranges (in radians)
    az_cent_deg, az_half_deg = azimuthal_deg_range
    pol_cent_deg, pol_half_deg = polar_deg_range
    az_cent_rad, az_half_rad = np.deg2rad([az_cent_deg, az_half_deg])
    pol_cent_rad, pol_half_rad = np.deg2rad([pol_cent_deg, pol_half_deg])

    azimuthal_angles = np.linspace(
        az_cent_rad - az_half_rad,
        az_cent_rad + az_half_rad,
        initialization_steps,
    )
    polar_angles = np.linspace(
        pol_cent_rad - pol_half_rad,
        pol_cent_rad + pol_half_rad,
        initialization_steps,
    )

    # Create grid of angles
    theta_grid, phi_grid = np.meshgrid(
        azimuthal_angles,
        polar_angles,
        indexing="ij",
    )  # both shape (N, N)

    # Compute normals for each (θ, φ)
    n = geometry.generate_normal(theta_grid, phi_grid)  # shape (N, N, 3)

    # Compute offsets L_{ij} = n_{ij} · com_phy
    L_grid = np.tensordot(n, com_phy, axes=([-1], [0]))  # shape (N, N)

    # Flatten into list of plane parameters
    planes = np.stack(
        [
            theta_grid.ravel(),
            phi_grid.ravel(),
            L_grid.ravel(),
        ],
        axis=1,
    )  # shape (N*N, 3)

    # Define the ranges for normalization
    theta_range = 2 * az_half_rad
    phi_range = 2 * pol_half_rad
    L_range = np.max(L_grid) - np.min(L_grid)

    print(f"Starting robustness test with {len(planes)} initialization points...")
    print(
        f"Angular ranges: θ ∈ [{np.rad2deg(az_cent_rad - az_half_rad):.1f}°, {np.rad2deg(az_cent_rad + az_half_rad):.1f}°], "
        f"φ ∈ [{np.rad2deg(pol_cent_rad - pol_half_rad):.1f}°, {np.rad2deg(pol_cent_rad + pol_half_rad):.1f}°]",
    )

    # Use multiprocessing for parallel optimization
    n_processes = min(
        multiprocessing.cpu_count(),
        len(planes),
    )  # Don't use more processes than initialization points
    print(f"Using {n_processes} parallel processes for optimization...")

    # Prepare arguments for multiprocessing
    task_args = []
    for index, p in enumerate(planes):
        task_args.append(
            (
                index,
                p,
                bone,
                interpolator_intensity,
                voxel_size,
                optimal_plane_params,
                theta_range,
                phi_range,
                L_range,
            ),
        )

    # Run optimizations in parallel
    with multiprocessing.Pool(processes=n_processes) as pool:
        optimization_results = pool.map(_optimize_single_initialization, task_args)

    # Sort results by index to maintain order
    optimization_results.sort(key=lambda x: x["index"])

    # Extract error metrics and print results
    error_metrics = []
    print("\nOptimization Results:")
    for result in optimization_results:
        error_metrics.append(result["error_metric"])

        # Print result summary
        idx = result["index"]
        initial_p = result["initial_params"]
        success = result["success"]

        print(
            f"Init {idx + 1}/{len(planes)}: θ={np.rad2deg(initial_p[0]):.1f}°, φ={np.rad2deg(initial_p[1]):.1f}°, "
            f"L={initial_p[2]:.2f}, Obj={result['initial_objective']:.2e}, Delta={result['delta']:.2e}",
        )

        if success and result["optimized_params"] is not None:
            opt_p = result["optimized_params"]
            print(
                f"Optimized {idx + 1}/{len(planes)}: θ={np.rad2deg(opt_p[0]):.1f}°, "
                f"φ={np.rad2deg(opt_p[1]):.1f}°, L={opt_p[2]:.2f}, "
                f"Obj={result['optimized_objective']:.2e}, Error={result['error_metric']:.4f}",
            )
        else:
            error_msg = result.get("error", "optimization failed")
            print(f"Failed {idx + 1}/{len(planes)}: {error_msg}")

    # Remove index key from results as it's no longer needed
    for result in optimization_results:
        del result["index"]

    # Convert to numpy arrays for easier handling
    error_metrics = np.array(error_metrics)

    # Compute summary statistics
    finite_errors = error_metrics[np.isfinite(error_metrics)]
    if len(finite_errors) > 0:
        min_error = np.min(finite_errors)
        max_error = np.max(finite_errors)
        mean_error = np.mean(finite_errors)
        std_error = np.std(finite_errors)
        success_rate = len(finite_errors) / len(error_metrics)

        # Calculate percentage of samples with approximately zero error
        # Use a small tolerance to account for numerical precision
        zero_error_tolerance = 1e-6
        zero_error_count = np.sum(finite_errors < zero_error_tolerance)
        zero_error_percentage = (zero_error_count / len(error_metrics)) * 100

    else:
        min_error = max_error = mean_error = std_error = np.inf
        success_rate = 0.0
        zero_error_count = 0
        zero_error_percentage = 0.0

    elapsed_time = time.time() - start_time

    print(f"Robustness test completed in {elapsed_time:.2f} seconds")
    print(f"Success rate: {success_rate:.1%}")
    print(
        f"Error statistics: min={min_error:.4f}, max={max_error:.4f}, "
        f"mean={mean_error:.4f}±{std_error:.4f}",
    )
    print(
        f"Near-zero error samples: {zero_error_count}/{len(error_metrics)} ({zero_error_percentage:.1f}%)",
    )

    # Prepare return dictionary
    results = {
        "optimization_results": optimization_results,
        "error_metrics": error_metrics,
        "azimuthal_angles": azimuthal_angles,
        "polar_angles": polar_angles,
        "theta_grid": theta_grid,
        "phi_grid": phi_grid,
        "L_grid": L_grid,
        "optimal_plane_params": optimal_plane_params,
        "statistics": {
            "min_error": min_error,
            "max_error": max_error,
            "mean_error": mean_error,
            "std_error": std_error,
            "success_rate": success_rate,
            "zero_error_count": zero_error_count,
            "zero_error_percentage": zero_error_percentage,
            "elapsed_time": elapsed_time,
        },
        "ranges": {
            "theta_range": theta_range,
            "phi_range": phi_range,
            "L_range": L_range,
        },
        "parameters": {
            "azimuthal_deg_range": azimuthal_deg_range,
            "polar_deg_range": polar_deg_range,
            "initialization_steps": initialization_steps,
        },
    }

    # Save results to files for later analysis
    if save_results:
        print("Saving robustness test results...")

        # Save main results as compressed numpy file
        np.savez_compressed(
            results_file,
            error_metrics=error_metrics,
            azimuthal_angles=azimuthal_angles,
            polar_angles=polar_angles,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
            L_grid=L_grid,
            optimal_plane_params=optimal_plane_params,
            # Convert optimization results to arrays for efficient storage
            initial_params=np.array(
                [r["initial_params"] for r in optimization_results],
            ),
            optimized_params=np.array(
                [
                    r["optimized_params"]
                    if r["optimized_params"] is not None
                    else np.full(3, np.nan)
                    for r in optimization_results
                ],
            ),
            initial_objectives=np.array(
                [r["initial_objective"] for r in optimization_results],
            ),
            optimized_objectives=np.array(
                [r["optimized_objective"] for r in optimization_results],
            ),
            deltas=np.array([r["delta"] for r in optimization_results]),
            success_flags=np.array([r["success"] for r in optimization_results]),
            nit_counts=np.array([r["nit"] for r in optimization_results]),
        )

        # Save metadata and parameters as JSON for human readability
        import json

        # Convert NumPy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        metadata = {
            "parameters": convert_numpy_types(results["parameters"]),
            "statistics": convert_numpy_types(results["statistics"]),
            "ranges": convert_numpy_types(results["ranges"]),
            "grid_shape": [len(azimuthal_angles), len(polar_angles)],
            "total_optimizations": len(optimization_results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print("Results saved to:")
        print(f"  - {results_file}")
        print(f"  - {metadata_file}")

    # Generate and save heatmap of error metrics
    print("Generating robustness heatmap...")
    visualization.plot_robustness_heatmap(
        azimuthal_angles=azimuthal_angles,
        polar_angles=polar_angles,
        error_array=error_metrics,
        title="Robustness Test - Parameter Error vs Initialization",
        output_path=output_path,
    )

    return results


def optimize_plane(
    initial_params_array,
    image,
    interpolator_intensity,
    voxel_size,
    delta,
):
    """Optimize the plane parameters to minimize the objective function using BFGS.

    Parameters
    ----------
    initial_params_array : array-like of shape (3,)
        Initial guess for plane parameters [azimuthal, polar, L].
    image : np.ndarray
        The image or mask to optimize over.
    interpolator_intensity : RegularGridInterpolator
        Interpolator for the original image.
    voxel_size : array-like of shape (3,)
        Physical voxel size for each axis.
    delta : float
        Huber loss delta parameter.

    Returns
    -------
    OptimizeResult
        Result object from scipy.optimize.minimize, with additional attributes:
        - objective_value_list: list of objective values per iteration.
        - params_list: list of parameter arrays per iteration.

    """
    objective_value_list = []
    params_list = []

    def callback(xk):
        f_val = compute_objective(xk, image, interpolator_intensity, voxel_size, delta)
        objective_value_list.append(f_val)
        params_list.append(xk.copy())

    res = sp.optimize.minimize(
        compute_objective,
        x0=initial_params_array,
        args=(image, interpolator_intensity, voxel_size, delta),
        method="BFGS",
        jac=None,
        callback=callback,
    )
    params_list.append(res.x.copy())
    objective_value_list.append(res.fun)

    res.objective_value_list = objective_value_list
    res.params_list = params_list

    return res


def run_or_load_optimization(
    output_path_patient,
    image,
    voxel_size_image,
    delta,
    initial_plane,
    bone_ct,
    interpolator,
    optimized_parameter_list,
    optimized_objective_value_list,
):
    """Run or load the optimization of plane parameters for a single patient.

    Parameters
    ----------
    output_path_patient : str
        Directory to save/load optimization results.
    image : np.ndarray
        The original image array.
    voxel_size_image : array-like of shape (3,)
        Physical voxel size for each axis.
    delta : float
        Huber loss delta parameter.
    initial_plane : array-like of shape (3,)
        Initial guess for plane parameters.
    bone_ct : np.ndarray
        Bone mask or intensity image.
    interpolator : RegularGridInterpolator
        Interpolator for the original image.
    optimized_parameter_list : list
        List to append optimized parameters to.
    optimized_objective_value_list : list
        List to append optimized objective values to.

    Returns
    -------
    tuple
        optimized_parameters : np.ndarray
            Optimized plane parameters [azimuthal, polar, L].
        optimized_objective_value : float
            Final objective function value.

    """
    param_path = os.path.join(output_path_patient, "parameter_array.npy")
    obj_path = os.path.join(output_path_patient, "objective_value_array.npy")

    if os.path.exists(param_path) and os.path.exists(obj_path):
        # load the last saved result
        params_arr = np.load(param_path)
        obj_arr = np.load(obj_path)
        optimized_parameters = params_arr[-1]
        optimized_objective_value = obj_arr[-1]
        print(
            f"Optimized parameters: "
            f"{np.rad2deg(optimized_parameters[0]):.2f}°, "
            f"{np.rad2deg(optimized_parameters[1]):.2f}°, "
            f"{optimized_parameters[2]:.2f} mm "
            f"with MSE {optimized_objective_value:.2f}",
        )

        visualization.make_plane_gif(
            image,
            voxel_size_image,
            params_arr,
            obj_arr,
            output_path_patient,
        )
        visualization.make_optimization_gif(
            obj_arr,
            output_path_patient,
            title="Optimization of Objective Function",
        )
    else:
        start = time.time()
        res = optimize_plane(
            initial_plane,
            bone_ct,
            interpolator,
            voxel_size_image,
            delta,
        )
        end = time.time()
        print(f"Optimization took {end - start:.2f} seconds.")

        optimized_parameters = res.x
        optimized_objective_value = res.fun
        print(
            f"Optimized parameters: "
            f"θ = {np.rad2deg(optimized_parameters[0]):.2f}°, "
            f"φ = {np.rad2deg(optimized_parameters[1]):.2f}°, "
            f"L = {optimized_parameters[2]:.2f} mm "
            f"with MSE {optimized_objective_value:.2f}",
        )

        visualization.make_plane_gif(
            image,
            voxel_size_image,
            np.array(res.params_list),
            np.array(res.objective_value_list),
            output_path_patient,
        )
        visualization.make_optimization_gif(
            np.array(res.objective_value_list),
            output_path_patient,
            title="Optimization of Objective Function",
        )
        # np.save(param_path,   np.array([[-0.06283185307179585, 1.6244956418729801, 262]]))
        # np.save(obj_path,     np.array(res.objective_value_list))

        # optimized_parameters_manual_10587029 = (-0.06283185307179585, 1.6244956418729801, 262)
        # optimized_parameters_manual_10774767 = (-0.0017453292519943233, 1.594660423674561, 240)
        # optimized_parameters_manual_10780163 = (-0.03490658503988659, 1.5982842763770915, 269)

        np.save(param_path, np.array(res.params_list))
        np.save(obj_path, np.array(res.objective_value_list))

        optimized_parameter_list.append(optimized_parameters)
        optimized_objective_value_list.append(optimized_objective_value)

    return optimized_parameters, optimized_objective_value
