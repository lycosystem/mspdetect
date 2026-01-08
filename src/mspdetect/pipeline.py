"""This module defines the pipeline for detecting the MSP of many CT images."""

import os
import time
from typing import Any

import numpy as np
import tqdm
import yaml
from loguru import logger

from mspdetect import io, msp_widget, optimization, preprocessing


def load_config(config_path="../config.yaml") -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_patient_folders(
        base_path: str,
        patient: str | list[str] | None
    ) -> list[str]:
    """Get list of patient folders to process."""
    if patient is None:
        # Process all patients in base_path
        return sorted([d for d in os.listdir(base_path)])
    elif isinstance(patient, list):
        # Process specific list of patients
        return patient
    elif os.path.isfile(patient):
        # Read patient list from file
        with open(patient) as f:
            return [line.strip() for line in f if line.strip()]
    else:
        # Process single patient
        return [str(patient)]

def generate_output_directory_name(base_name, pipeline_config, opt_config):
    """Generate output directory name encoding all parameters.

    Parameters
    ----------
    base_name : str
        Base name for output directory (e.g., "Patients_processed")
    pipeline_config : dict
        Cropping parameters: margins, reference, or slice_range
    opt_config : dict
        Optimization parameters: azimuthal_range, polar_range, initialization_method

    Returns
    -------
    str
        Directory name with encoded parameters

    Examples
    --------
    >>> generate_output_directory_name(
    ...     "Patients_processed",
    ...     {"margins": 20, "reference": "GTVp"},
    ...     {"azimuthal_range": [-180, 180], "polar_range": [0, 180], "initialization_method": "grid"}
    ... )
    'Patients_processed_m20_refGTVp_az-180to180_pol0to180_initgrid'

    """
    components = [base_name]

    # Crop configuration
    if "slice_range" in pipeline_config and pipeline_config["slice_range"] is not None:
        start, end = pipeline_config["slice_range"]
        components.append(f"range{start}-{end}")
    elif "fixed_size" in pipeline_config and pipeline_config["fixed_size"] is not None:
        fixed_size = pipeline_config["fixed_size"]
        components.append(f"fixed{fixed_size}vox")
    elif "margins" in pipeline_config and pipeline_config["margins"] is not None:
        margins = pipeline_config["margins"]
        if isinstance(margins, int):
            components.append(f"m{margins}")
        elif isinstance(margins, tuple):
            lower, upper = margins
            components.append(f"m{lower}-{upper}")

        ref = pipeline_config.get("reference", "GTVp")
        components.append(f"ref{ref}")
    else:
        # No margins, just reference
        ref = pipeline_config.get("reference", "GTVp")
        components.append(f"ref{ref}")

    # Optimization ranges
    az_range = opt_config.get("azimuthal", [0, 90])
    pol_range = opt_config.get("polar", [90, 10])
    components.append(f"az{az_range[0]}to{az_range[1]}")
    components.append(f"pol{pol_range[0]}to{pol_range[1]}")

    # Initialization method
    init_steps = opt_config.get("initialization_steps", "grid")
    components.append(f"init{init_steps}")

    return "_".join(components)


def _display_full_anatomy(
    struct_dict: dict,
    voxel_size: np.ndarray,
    HU_range: list[int],
) -> None:
    """Display full anatomy for verification."""
    image_orig = struct_dict["Image"][0]
    bone_ct_orig = np.clip(image_orig, HU_range[0], HU_range[1])
    struct_dict["Image"] = (bone_ct_orig, voxel_size)

    msp_widget.display_scrollable_views(
        struct_dict,
        voxel_size,
        plane_coeffs_list=None,
        optimization_methods_list=None,
    )


def _run_preprocessing_pipeline(
    struct_dict: dict,
    pipeline_config: dict[str, Any],
    output_path: str,
    voxel_size: np.ndarray,
) -> dict[str, Any]:
    """Run the preprocessing pipeline."""
    # Crop volumes
    (
        struct_dict_cropped,
        interpolation_image,
        interpolation_pad,
        crop_start,
        crop_end,
        crop_axis,
    ) = preprocessing.crop_patient_volumes(
        struct_dict,
        slice_axis=pipeline_config["slice_axis"],
        slice_range=pipeline_config["slice_range"],
        margins=pipeline_config["margins"],
        reference=pipeline_config["reference"],
        crop_mode=pipeline_config["crop_mode"],
        fixed_size=pipeline_config["fixed_size"],
    )

    # Preprocess bone image
    image, bone_ct = preprocessing.preprocess_bone_image(
        struct_dict_cropped,
        HU_range=pipeline_config["HU_range"],
    )

    # Get interpolator
    interpolator = preprocessing.get_cached_interpolator(
        output_path,
        interpolation_image,
        voxel_size,
        pad_slices=interpolation_pad,
    )

    return {
        "struct_dict_cropped": struct_dict_cropped,
        "image": image,
        "bone_ct": bone_ct,
        "interpolator": interpolator,
        "crop_start": crop_start,
        "crop_end": crop_end,
        "margins": pipeline_config["margins"],
        "reference": pipeline_config["reference"],
    }


def _run_optimization_pipeline(
    processed_data: dict[str, Any],
    optimization_config: dict[str, Any],
    output_path: str,
    voxel_size: np.ndarray,
) -> dict[str, Any]:
    """Run the optimization pipeline."""
    # Parameter initialization
    initial_plane, delta = optimization.parameter_initialization(
        processed_data["image"],
        processed_data["bone_ct"],
        output_path,
        processed_data["interpolator"],
        voxel_size,
        azimuthal_deg_range=optimization_config["azimuthal"],
        polar_deg_range=optimization_config["polar"],
        initialization_steps=optimization_config["initialization_steps"],
    )

    # Run optimization
    optimized_parameters, optimized_objective_values = (
        optimization.run_or_load_optimization(
            output_path,
            processed_data,
            voxel_size,
            delta,
            initial_plane,
            [],  # optimized_parameter_list
            [],  # optimized_objective_value_list
        )
    )

    return {
        "parameters": optimized_parameters,
        "objective_value": optimized_objective_values,
    }


def _run_visualization_pipeline(
    struct_dict_cropped: dict,
    voxel_size: np.ndarray,
    optimized_parameters: np.ndarray,
    visualization_config: dict[str, Any],
) -> None:
    """Run visualization components."""
    if visualization_config["verification_widget"]:
        msp_widget.display_scrollable_views(
            struct_dict_cropped,
            voxel_size,
            plane_coeffs_list=[optimized_parameters],
            optimization_methods_list=["MSP optimized"],
        )

    if visualization_config["parameter_tweaking"]:
        msp_widget.display_interactive_plane_widget(
            struct_dict=struct_dict_cropped,
            voxel_size=voxel_size,
            initial_plane_params=optimized_parameters,
            theta_range=(-10, 10),
            phi_range=(80, 100),
            L_range=(200, 300),
        )


def _save_batch_results(output_path: str, results: dict[str, list]) -> None:
    """Save batch processing results."""
    np.save(
        os.path.join(output_path, "optimized_parameters.npy"),
        np.array(results["parameters"]),
    )
    np.save(
        os.path.join(output_path, "objective_values.npy"),
        np.array(results["objective_values"]),
    )


def process_single_patient(
    folder_path: str,
    output_path: str,
    pipeline_config: dict[str, Any],
    optimization_config: dict[str, Any],
    visualization_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Process a single patient through the MSP detection pipeline."""
    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    structure_name_mapping = pipeline_config.get("structure_name_mapping", None)

    # Load patient data
    struct_dict = io.load_patient_structures(
        folder_path,
        pipeline_config["structure_names"],
        structure_name_mapping,
    )

    if struct_dict is None:
        logger.warning(f"No matching files found in {folder_path}. Skipping...")
        return None

    voxel_size = struct_dict["Image"][1]

    # Optional: Check full anatomy
    if pipeline_config["check_full_anatomy"]:
        _display_full_anatomy(struct_dict, voxel_size, pipeline_config["HU_range"])

    # Preprocessing pipeline
    processed_data = _run_preprocessing_pipeline(
        struct_dict,
        pipeline_config,
        output_path,
        voxel_size,
    )

    # Optimization pipeline
    optimization_results = _run_optimization_pipeline(
        processed_data,
        optimization_config,
        output_path,
        voxel_size,
    )

    # Visualization pipeline
    if any(visualization_config.values()):
        _run_visualization_pipeline(
            processed_data["struct_dict_cropped"],
            voxel_size,
            optimization_results["parameters"],
            visualization_config,
        )

    return optimization_results


def run_msp_pipeline(
    config_path: str | None,
) -> None:
    """Run the MSP detection pipeline with configuration file.

    Parameters
    ----------
    config_path : str, optional
        Path to user configuration file (TOML format)

    """
    # Load and validate configuration
    config = load_config(config_path)

    # Extract configuration sections
    base_path = config["base_path"]
    # output_path = config["output_path"]
    patient = config["patient"]

    pipeline_config = config["pipeline"]
    optimization_config = config["optimization"]
    visualization_config = config["visualization"]

    output_path = generate_output_directory_name(
        base_path,
        pipeline_config,
        optimization_config,
    )
    start_time = time.time()
    results = {"parameters": [], "objective_values": []}

    # Get patient folders
    folders = _get_patient_folders(base_path, patient)

    for folder in tqdm.tqdm(folders, desc="Processing CTs"):
        logger.info(f"Processing patient: {folder}")

        patient_results = process_single_patient(
            folder_path=os.path.join(base_path, folder),
            output_path=os.path.join(output_path, folder),
            pipeline_config=pipeline_config,
            optimization_config=optimization_config,
            visualization_config=visualization_config,
        )

        if patient_results:
            results["parameters"].append(patient_results["parameters"])
            results["objective_values"].append(patient_results["objective_value"])

    end_time = time.time()
    duration = end_time - start_time

    # Save batch results
    _save_batch_results(output_path, results)
    logger.info(
        f"Pipeline completed for {len(folders)} patient(s) in {duration:.2f} seconds.",
    )
