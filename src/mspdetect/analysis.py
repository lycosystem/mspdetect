import os
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

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


def compute_volumes_for_structures(plane_params, struct_dict, voxel_size, structures):
    """Computes the volumes on the left and right side of the detected mid-sagittal plane
    for the specified structures.
    """
    results = {}

    for structure in structures:
        if structure in struct_dict:
            if isinstance(struct_dict[structure], tuple):
                vol_left, vol_right = compute_volume_structure(
                    plane_params,
                    struct_dict[structure][0],
                    voxel_size,
                )

            elif isinstance(struct_dict[structure], np.ndarray):
                vol_left, vol_right = compute_volume_structure(
                    plane_params,
                    struct_dict[structure],
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


def pipeline_volumes_computation(
    config_path: str | None,
    results_path: Path,
    structure_names: list[str] | None = ["Mandible", "Spinal Cord", "Body"],
    crop_level: str | None = "full",
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
    output_path = config["output_path"]
    patient = config["patient"]

    pipeline_config = config["pipeline"]

    folders = msp.pipeline._get_patient_folders(base_path, patient)

    volumes = {}

    for folder in tqdm.tqdm(folders, desc="Patient Volumes"):
        folder_path = os.path.join(base_path, folder)
        result_path = os.path.join(output_path, folder)

        # load structures of patient
        struct_dict = msp.io.load_patient_structures(
            folder_path,
            structure_names=["Image", "Spinal Cord", "GTVp", "Mandible", "Body"],
        )
        voxel_size = struct_dict["Image"][1]

        # load msp parameters
        params = load_plane_params(result_path)

        # compute cropped patient structure
        struct_dict_cropped, _, start, end, _ = msp.preprocessing.crop_patient_volumes(
            struct_dict,
            pipeline_config["slice_axis"],
            pipeline_config["slice_range"],
        )

        if crop_level == "full":
            patient_volumes = compute_volumes_for_structures(
                params,
                struct_dict_cropped,
                voxel_size,
                structure_names,
            )

        elif crop_level == "partial":
            struct_dict_cropped_mid, _, start_mid, end_mid, _ = (
                msp.preprocessing.crop_patient_volumes_mid(
                    struct_dict,
                    pipeline_config["slice_axis"],
                    pipeline_config["slice_range"],
                )
            )

            # calculate new parameters for partially cropped image
            azimuthal, polar, L = params
            n = msp.geometry.generate_normal(azimuthal, polar)
            print(voxel_size[2])
            k = np.array([0.0, 0.0, (start - start_mid) * voxel_size[2]])
            new_L = L + np.dot(k, n)

            params_new = np.array([azimuthal, polar, new_L])
            patient_volumes = compute_volumes_for_structures(
                params_new,
                struct_dict_cropped_mid,
                voxel_size,
                ["Mandible", "Spinal Cord", "Body"],
            )

        elif crop_level == "none":
            # calculate new parameters for none cropped image
            azimuthal, polar, L = params
            n = msp.geometry.generate_normal(azimuthal, polar)
            print(voxel_size[2])
            k = np.array([0.0, 0.0, start * voxel_size[2]])
            new_L = L + np.dot(k, n)

            params_new = np.array([azimuthal, polar, new_L])
            patient_volumes = compute_volumes_for_structures(
                params_new,
                struct_dict,
                voxel_size,
                ["Mandible", "Spinal Cord", "Body"],
            )

        else:
            raise Exception(
                f"Invalid crop level: {crop_level}! Must be either none, partial or full.",
            )

        volumes[folder] = patient_volumes

        # Create DataFrame with columns
        df_volumes = pd.DataFrame.from_dict(volumes, orient="index")

        # Create MultiIndex columns
        df_volumes.columns = pd.MultiIndex.from_tuples(
            df_volumes.columns,
        )

        # Reorder columns for consistent presentation
        column_order = [
            (s, side)
            for s in ["Mandible", "Spinal Cord", "Body"]
            for side in ["left", "right"]
        ]
        df_volumes = df_volumes.reindex(columns=column_order)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{results_path}/volumes_crop_level={crop_level}_{timestamp}.csv"
        df_volumes.to_csv(output_file)
