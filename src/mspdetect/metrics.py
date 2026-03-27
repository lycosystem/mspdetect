import functools
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial import ConvexHull, distance
from tqdm import tqdm

from mspdetect import geometry, io, optimization, pipeline, preprocessing


def extract_metrics(
    distances,
    voxel_volume,
):
    if len(distances) > 0:
        mean = np.mean(distances)
        std = np.std(distances)
        maxval = np.max(distances)
        minval = np.min(distances)
        volume = len(distances) * voxel_volume
    else:
        mean, std, maxval, minval, volume = 0, 0, 0, 0, 0

    return {
        "volume": volume,
        "min": minval,
        "max": maxval,
        "mean": mean,
        "std": std,
    }


def load_patient_plane_params(result_path, struct_dict, pipeline_config, voxel_size):
    # Try to load full-image parameters
    full_param_path = os.path.join(result_path, "optimized_parameter_full_image.npy")

    if os.path.exists(full_param_path):
        return np.load(full_param_path, mmap_mode="r").copy()

    # Use cropped params and convert
    cropped_param_path = os.path.join(result_path, "parameter_array.npy")
    if not os.path.exists(cropped_param_path):
        raise FileNotFoundError(f"No parameter files found in {result_path}")

    params_cropped = np.load(cropped_param_path, mmap_mode="r")[-1].copy()

    struct_dict_cropped, _, crop_info = preprocessing.crop_patient_volumes(
        struct_dict,
        pipeline_config["slice_axis"],
        pipeline_config["slice_range"],
        pipeline_config["margins"],
        pipeline_config["reference"],
        pipeline_config["crop_mode"],
        pipeline_config["fixed_size"],
        pipeline_config.get("exclude_mandible", False),
        pipeline_config.get("crop_shoulders", None),
    )

    return geometry.crop_to_full_coords(
        params_cropped,
        crop_info,
        voxel_size,
    )


def get_max_distance(mask, voxel_size):
    points_idx = np.argwhere(mask)

    if points_idx.shape[0] < 2:
        return 0

    physical_points = points_idx * np.array(voxel_size)
    if physical_points.shape[0] > 2:
        try:
            hull = ConvexHull(physical_points)
            hull_points = physical_points[hull.vertices]
        except Exception:
            hull_points = physical_points
    else:
        hull_points = physical_points

    dists = distance.pdist(hull_points, metric="euclidean")
    max_dist = np.max(dists)

    return max_dist


def calculate_metrics_for_patient(
    patient_path,
    result_path,
    pipeline_config,
):
    structure_name_mapping = pipeline_config.get("structure_name_mapping")

    struct_dict = io.load_patient_structures(
        patient_path,
        structure_names=["GTVp", "Image"],
        structure_name_mapping=structure_name_mapping,
    )
    print(patient_path)

    if "GTVp" not in struct_dict:
        print("no GTVp")
        return None

    if not struct_dict:
        logger.warning(f"No matching files found in {patient_path}. Skipping...")
        return None

    voxel_size = struct_dict["Image"][1]

    # Try to load full-image parameters
    full_param_path = os.path.join(result_path, "optimized_parameter_full_image.npy")

    # if os.path.exists(full_param_path):
    #     params = np.load(full_param_path, mmap_mode="r").copy()
    # else:
    #     # Use cropped params and convert
    #     cropped_param_path = os.path.join(result_path, "parameter_array.npy")
    #     params_cropped = np.load(cropped_param_path, mmap_mode="r")[-1].copy()

    #     struct_dict_cropped, _, crop_info = preprocessing.crop_patient_volumes(
    #         struct_dict,
    #         pipeline_config["slice_axis"],
    #         pipeline_config["slice_range"],
    #         pipeline_config["margins"],
    #         pipeline_config["reference"],
    #         pipeline_config["crop_mode"],
    #         pipeline_config["fixed_size"],
    #         pipeline_config.get("exclude_mandible", False),
    #         pipeline_config.get("crop_shoulders", None),
    #     )

    #     params = geometry.crop_to_full_coords(
    #         params_cropped,
    #         crop_info,
    #         voxel_size,
    #     )

    params = load_patient_plane_params(
        result_path, struct_dict, pipeline_config, voxel_size
    )

    logger.info(f"loaded params {params}")

    # Compute distances
    d, normal, indices_coord_syst, indices_image = (
        optimization.compute_signed_distances(
            params,
            struct_dict["GTVp"][0],
            voxel_size,
        )
    )
    voxel_volume = np.prod(voxel_size)

    # dictionary to save results
    metrics_dict = {"info": {}}

    gtv_mask = struct_dict["GTVp"][0]
    max_dist = get_max_distance(gtv_mask, voxel_size)
    metrics_dict["info"]["max_dist"] = max_dist

    idx_left = np.nonzero(d > 0)
    idx_right = np.nonzero(d < 0)

    metrics_dict["right"] = extract_metrics(np.absolute(d[idx_right]), voxel_volume)
    metrics_dict["left"] = extract_metrics(
        np.absolute(d[idx_left]),
        voxel_volume,
    )

    # determine the side
    if metrics_dict["left"]["volume"] > metrics_dict["right"]["volume"]:
        side = "left"
    elif metrics_dict["right"]["volume"] > metrics_dict["left"]["volume"]:
        side = "right"
    else:
        side = "central"
    metrics_dict["info"]["side"] = side

    # calculate extension / distance to midline
    if (metrics_dict["left"]["volume"] > 0) & (metrics_dict["right"]["volume"] > 0):
        metrics_dict["info"]["extension"] = True
        if side in ["right", "central"]:
            distance_msp = metrics_dict["left"]["max"]
        else:
            distance_msp = metrics_dict["right"]["max"]
        max_dist_sagittal = metrics_dict["left"]["max"] + metrics_dict["right"]["max"]
    else:
        metrics_dict["info"]["extension"] = False
        distance_msp = -metrics_dict[side]["min"]
        max_dist_sagittal = metrics_dict[side]["max"] - metrics_dict[side]["min"]
    metrics_dict["info"]["distance_msp"] = distance_msp
    metrics_dict["info"]["max_dist_sagittal"] = max_dist_sagittal

    return metrics_dict


def _process_patient_wrapper(
    row_iter,
    base_path,
    output_path,
    pipeline_config,
):
    _, row = row_iter
    try:
        # get patient path
        patient_path = base_path / row["dataset"] / str(row["patient"])
        # get result path
        result_path = output_path / row["dataset"] / str(row["patient"])
        patient_metrics = calculate_metrics_for_patient(
            patient_path,
            result_path,
            pipeline_config,
        )
        return (row["patient"], patient_metrics)
    except Exception as e:
        patient = row["patient"]
        logger.error(f"Error processing {patient}: {e}")
        return (row["patient"], None)


def pipeline_metrics_computation(
    config_path: str | None,
    log_file: Path | None,
    base_path: Path | None = None,
    output_path: Path | None = None,
    result_path: Path | None = None,
):
    config = pipeline.load_config(config_path)

    pipeline_config = config["pipeline"]

    log_csv = pd.read_csv(log_file, header=[0])

    if base_path is None:
        base_path = Path(
            "/home/noemibuhrer/Datasets/RADCURE/DATASETS_NII"
        )  # Update this to your actual base path
    if output_path is None:
        output_path = Path(
            "/home/noemibuhrer/Datasets/RADCURE/msp_results/final"
        )  # Directory where files will be organized
    result_dict = {}
    num_processes = 4

    process_func = functools.partial(
        _process_patient_wrapper,
        base_path=base_path,
        output_path=output_path,
        pipeline_config=pipeline_config,
    )
    # for patient_id, row in tqdm(log_csv.iterrows(), total=len(log_csv)):
    #     # get patient path
    #     patient_path = base_source_path / row["dataset"] / row["patient"]
    #     # get result path
    #     result_path = output_base_path / row["dataset"] / row["patient"]

    #     patient_dict = calculate_metrics_for_patient(
    #         patient_path,
    #         result_path,
    #         pipeline_config
    #     )
    #     if patient_dict:
    #         patient_dict["info"]["patient"] = row["patient"]
    #     else:
    #         patient_dict = {"info": {"patient": row["patient"]}}

    #     multilevel_dict = {(level2, level3): value
    #                             for level2, inner_inner_dict in patient_dict.items()
    #                             for level3, value in inner_inner_dict.items()}

    #     result_dict[patient_id] = multilevel_dict
    if num_processes == 1:
        for row_iter in tqdm(
            log_csv.iterrows(), total=len(log_csv), desc="Extract metrics"
        ):
            patient, patient_dict = process_func(row_iter)
            if patient_dict:
                patient_dict["info"]["patient"] = patient
            else:
                patient_dict = {"info": {"patient": patient}}
            multilevel_dict = {
                (level2, level3): value
                for level2, inner_inner_dict in patient_dict.items()
                for level3, value in inner_inner_dict.items()
            }

            result_dict[patient] = multilevel_dict
    else:
        with Pool(processes=num_processes) as pool:
            results = list(
                tqdm(
                    pool.imap(process_func, log_csv.iterrows()),
                    total=len(log_csv),
                    desc="Extract metrics",
                )
            )

        for patient, patient_dict in results:
            if patient_dict:
                patient_dict["info"]["patient"] = patient
            else:
                patient_dict = {"info": {"patient": patient}}
            multilevel_dict = {
                (level2, level3): value
                for level2, inner_inner_dict in patient_dict.items()
                for level3, value in inner_inner_dict.items()
            }

            result_dict[patient] = multilevel_dict

    df = pd.DataFrame.from_dict(result_dict, orient="index")
    df.columns = pd.MultiIndex.from_tuples(
        df.columns,
    )
    # save csv
    # df.to_csv('/home/noemibuhrer/Projects/RADCURE-analysis/msp_pipeline/logs/metrics.csv', index=False)
    df.to_csv(result_path, index=False)
