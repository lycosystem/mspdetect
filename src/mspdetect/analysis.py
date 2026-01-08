import os
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from loguru import logger

import mspdetect as msp


def load_plane_params(path: Path):
    path_to_file = os.path.join(path, "parameter_array.npy")
    params = np.load(path_to_file, mmap_mode="r")[-1].copy()
    return params


def compute_volume_structure(plane_params, structure, voxel_size):
    d, _, _, _ = msp.optimization.compute_signed_distances(
        plane_params,
        structure,
        voxel_size,
    )
    voxel_volume = np.prod(voxel_size)
    vol_left = np.count_nonzero(d > 0) * voxel_volume
    vol_right = np.count_nonzero(d < 0) * voxel_volume

    return vol_left, vol_right


def compute_volumes_for_structures(
    struct_dict,
    structures,
    plane_params,
    voxel_size,
    reference_structure,
    margins,
    slice_range,
    slice_axis,
):
    """Computes the volumes on the left and right side of the detected mid-sagittal plane
    for the specified structures.
    """
    results = {}

    struct_dict_cropped, _, _, crop_start, crop_end, crop_axis = (
        msp.preprocessing.crop_patient_volumes(
            struct_dict,
            slice_axis,
            slice_range,
            margins,
            reference_structure,
        )
    )
    plane_params_cropped = msp.geometry.full_to_crop_coords(
        plane_params,
        crop_start,
        voxel_size,
    )
    print("cropped_params = ", plane_params_cropped)
    for structure in structures:
        if structure in struct_dict_cropped:
            if isinstance(struct_dict_cropped[structure], tuple):
                vol_left, vol_right = compute_volume_structure(
                    plane_params_cropped,
                    struct_dict_cropped[structure][0],
                    voxel_size,
                )

            elif isinstance(struct_dict_cropped[structure], np.ndarray):
                vol_left, vol_right = compute_volume_structure(
                    plane_params_cropped,
                    struct_dict_cropped[structure],
                    voxel_size,
                )
            else:
                raise Exception("Neither tuple nor array!")
            # Use tuples as keys for hierarchical columns
            results[(structure, "left")] = vol_left
            results[(structure, "right")] = vol_right
        else:
            # Add NaN for missing structures (better than 0 for analysis)
            results[(structure, "left")] = np.nan
            results[(structure, "right")] = np.nan

    return results


def calculate_volumes_for_patient(
    patient_path,
    result_path,
    pipeline_config,
    structures,
    crop_range,
    crop_reference,
    crop_margin,
    crop_slice,
):
    """Calculate volumes for one patient using appropriate coordinate system."""
    structure_name_mapping = pipeline_config.get("structure_name_mapping", None)

    structures_load = structures.copy()
    if "GTVp" not in structures_load:
        structures_load.extend(["GTVp"])
    if "Image" not in structures_load:
        structures_load.extend(["Image"])

    # load structures of patient
    struct_dict = msp.io.load_patient_structures(
        patient_path,
        # structure_names=["Image", "Spinal Cord", "GTVp", "Mandible", "Body"],
        structure_names=structures_load,
        structure_name_mapping=structure_name_mapping,
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

        struct_dict_cropped, _, _, crop_start, crop_end, _ = (
            msp.preprocessing.crop_patient_volumes(
                struct_dict,
                pipeline_config["slice_axis"],
                pipeline_config["slice_range"],
                pipeline_config["margins"],
                pipeline_config["reference"],
                pipeline_config["crop_mode"],
                pipeline_config["fixed_size"],
            )
        )
        params = msp.geometry.crop_to_full_coords(
            params_cropped,
            crop_start,
            voxel_size,
        )

    logger.info(f"loaded params {params}")
    volumes = compute_volumes_for_structures(
        struct_dict,
        structures,
        params,
        voxel_size,
        crop_reference,
        crop_margin,
        crop_range,
        crop_slice,
    )
    # print(volumes)
    return volumes


def pipeline_volumes_computation(
    config_path: str | None,
    # results_path: Path,
    structure_names: list[str] | None = ["Mandible", "Spinal Cord", "Body"],
    crop_range: tuple | None = None,
    crop_reference: str | None = "GTVp",
    crop_margin: int | None = 0,
    crop_slice: int | None = 2,
    # crop_level: str | None = "full",
) -> None:
    """Run volume calculation with the configuration file

    Parameters
    ----------
    crop_level: str
        - "full": crop tightly around GTVp
        - "partial": extended cropped image
        - "none": use entire image

    """
    # Load configuration
    config = msp.pipeline.load_config(config_path)

    # Extract configuration sections
    base_path = config["base_path"]
    patient = config["patient"]

    pipeline_config = config["pipeline"]
    optimization_config = config["optimization"]
    output_path = msp.pipeline.generate_output_directory_name(
        base_path,
        pipeline_config,
        optimization_config,
    )

    folders = msp.pipeline._get_patient_folders(base_path, patient)

    volumes = {}

    for folder in tqdm.tqdm(folders, desc="Patient Volumes"):
        folder_path = os.path.join(base_path, folder)
        result_path = os.path.join(output_path, folder)

        patient_result = calculate_volumes_for_patient(
            folder_path,
            result_path,
            pipeline_config,
            structure_names,
            crop_range,
            crop_reference,
            crop_margin,
            crop_slice,
        )
        if patient_result:
            volumes[folder] = patient_result

        # # load structures of patient
        # struct_dict = msp.io.load_patient_structures(
        #     folder_path,
        #     structure_names=["Image", "Spinal Cord", "GTVp", "Mandible", "Body"],
        # )
        # voxel_size = struct_dict["Image"][1]

        # # load msp parameters
        # params = load_plane_params(result_path)

        # # compute cropped patient structure
        # struct_dict_cropped, _, _, start, end, _ = msp.preprocessing.crop_patient_volumes(
        #     struct_dict,
        #     pipeline_config["slice_axis"],
        #     pipeline_config["slice_range"],
        # )

        # if crop_level == "full":
        #     patient_volumes = compute_volumes_for_structures(
        #         params,
        #         struct_dict_cropped,
        #         voxel_size,
        #         structure_names,
        #     )

        # elif crop_level == "partial":
        #     struct_dict_cropped_mid, _, _, start_mid, end_mid, _ = (
        #         msp.preprocessing.crop_patient_volumes_new(
        #             struct_dict,
        #             pipeline_config["slice_axis"],
        #             pipeline_config["slice_range"],
        #             margins = 20,
        #         )
        #     )

        #     # calculate new parameters for partially cropped image
        #     azimuthal, polar, L = params
        #     # n = msp.geometry.generate_normal(azimuthal, polar)
        #     # #print(voxel_size[2])
        #     # k = np.array([0.0, 0.0, (start - start_mid) * voxel_size[2]])
        #     # new_L = L + np.dot(k, n)

        #     #params_new = np.array([azimuthal, polar, new_L])
        #     patient_volumes = compute_volumes_for_structures(
        #         #params_new,
        #         params,
        #         struct_dict_cropped_mid,
        #         voxel_size,
        #         ["Mandible", "Spinal Cord", "Body"],
        #     )

        # elif crop_level == "none":
        #     # calculate new parameters for none cropped image
        #     azimuthal, polar, L = params
        #     n = msp.geometry.generate_normal(azimuthal, polar)
        #     print(voxel_size[2])
        #     k = np.array([0.0, 0.0, start * voxel_size[2]])
        #     new_L = L + np.dot(k, n)

        #     params_new = np.array([azimuthal, polar, new_L])
        #     patient_volumes = compute_volumes_for_structures(
        #         params_new,
        #         struct_dict,
        #         voxel_size,
        #         ["Mandible", "Spinal Cord", "Body"],
        #     )

        # else:
        #     raise Exception(
        #         f"Invalid crop level: {crop_level}! Must be either none, partial or full.",
        #     )

        # volumes[folder] = patient_volumes

    # Create DataFrame with columns
    df_volumes = pd.DataFrame.from_dict(volumes, orient="index")

    # Create MultiIndex columns
    df_volumes.columns = pd.MultiIndex.from_tuples(
        df_volumes.columns,
    )

    # Reorder columns for consistent presentation
    column_order = [
        (s, side)
        for s in ["Body", "Mandible", "Spinal Cord", "Larynx", "Lips", "Eye_L", "Eye_R"]
        for side in ["left", "right"]
    ]
    df_volumes = df_volumes.reindex(columns=column_order)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_path}/volumes_crop_reference={crop_reference}_crop_margin={crop_margin}_{timestamp}.csv"
    df_volumes.to_csv(output_file)
