"""This module defines the pipeline for detecting the MSP of many CT images."""

import os
import time
from typing import Any

import numpy as np
import yaml

from mspdetect import io, msp_widget, optimization, preprocessing, visualization


def load_config(config_path="../config.yaml") -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_patient_folders(base_path: str, patient: str | None) -> list[str]:
    """Get list of patient folders to process."""
    if patient is not None:
        return [str(patient)]
    return sorted([d for d in os.listdir(base_path)])


def _display_full_anatomy(
    struct_dict: dict, voxel_size: np.ndarray, HU_range: list[int],
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
    struct_dict_cropped, interpolation_image = preprocessing.crop_patient_volumes(
        struct_dict,
        slice_axis=pipeline_config["slice_axis"],
        slice_range=pipeline_config["slice_range"],
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
    )

    return {
        "struct_dict_cropped": struct_dict_cropped,
        "image": image,
        "bone_ct": bone_ct,
        "interpolator": interpolator,
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
            processed_data["image"],
            voxel_size,
            delta,
            initial_plane,
            processed_data["bone_ct"],
            processed_data["interpolator"],
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
        visualization.display_interactive_plane_widget(
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

    # Load patient data
    struct_dict = io.load_patient_structures(
        folder_path,
        pipeline_config["structure_names"],
    )

    if struct_dict is None:
        print(f"No matching files found in {folder_path}. Skipping...")
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
    output_path = config["output_path"]
    patient = config["patient"]

    pipeline_config = config["pipeline"]
    optimization_config = config["optimization"]
    visualization_config = config["visualization"]

    start_time = time.time()
    results = {"parameters": [], "objective_values": []}

    # Get patient folders
    folders = _get_patient_folders(base_path, patient)

    for folder in folders:
        print(f"Processing patient: {folder}")

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

        if patient is not None:
            break

    end_time = time.time()
    duration = end_time - start_time

    # Save batch results
    if patient is None:
        _save_batch_results(output_path, results)
        print(
            f"Pipeline completed for {len(folders)} patients in {duration:.2f} seconds.",
        )

    else:
        print(f"Pipeline completed for patient {patient} in {duration:.2f} seconds.")
