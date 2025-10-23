"""This module defines the pipeline for detecting the MSP of many CT images."""

import os
import time

import numpy as np

from mspdetect import io, msp_widget, optimization, preprocessing, visualization


def MSP_pipeline(
    base_path,
    output_path,
    structure_names=["Image"],
    slice_axis=2,
    HU_range=[300, 1500],
    slice_range=None,
    check_full_anatomy=False,
    azimuthal=(0, 90),
    polar=(90, 45),
    initialization_steps=10,
    verification_widget=False,
    parameter_tweaking=False,
    patient=None,
):
    """Complete pipeline for mid-sagittal plane detection across multiple patients.

    Parameters
    ----------
    base_path : str
        Path to the base directory containing patient folders.
    output_path : str
        Path to the directory where results will be saved.
    structure_names : list of str, optional
        List of structure names to load for each patient.
    slice_axis : int, optional
        Axis along which to crop (default: 2, axial).
    HU_range : list or tuple of (int, int), optional
        HU range for bone thresholding.
    slice_range : tuple or None, optional
        (start, end) indices for cropping. If None, uses GTVp mask or full range.
    azimuthal : tuple of (float, float), optional
        Center and half-width of azimuthal angle range in degrees.
    polar : tuple of (float, float), optional
        Center and half-width of polar angle range in degrees.
    initialization_steps : int, optional
        Number of steps for each angle in initialization.
    delta : float, optional
        Huber loss delta parameter.
    widget : bool, optional
        If True, display interactive widgets for visualization.

    Returns
    -------
    None

    """
    start_pipeline = time.time()
    optimized_parameter_list = []
    optimized_objective_value_list = []
    folders = sorted([d for d in os.listdir(base_path)], key=int)
    for folder in folders:
        if patient is not None:
            folder = str(patient)
        folder_path = os.path.join(base_path, folder)
        print(f"Processing folder: {folder_path}")
        output_path_patient = os.path.join(output_path, folder)
        os.makedirs(output_path_patient, exist_ok=True)
        struct_dict = io.load_patient_structures(folder_path, structure_names)
        voxel_size = struct_dict["Image"][1]

        if struct_dict is None:
            print(f"No matching files found in {folder_path}. Skipping...")
            continue

        if check_full_anatomy:
            image_orig = struct_dict["Image"][0]
            bone_ct_orig = np.clip(image_orig, HU_range[0], HU_range[1])
            struct_dict["Image"] = (bone_ct_orig, voxel_size)
            visualization.display_scrollable_views(
                struct_dict,
                voxel_size,
                plane_coeffs_list=None,
                optimization_methods_list=None,
            )

        # image_orig = struct_dict["Image"][0]
        # save_3d_slice_as_vector(image_orig, r"/home/loriskeller/Documents/Master Project/Results/Presentation",
        #                         {'axial': 125, 'coronal': 265, 'sagittal': 285},
        #                         0, 'image_original')

        # bone_ct_orig = np.clip(image_orig, HU_range[0], HU_range[1])
        # save_3d_slice_as_vector(bone_ct_orig, r"/home/loriskeller/Documents/Master Project/Results/Presentation",
        #                         {'axial': 125, 'coronal': 265, 'sagittal': 285},
        #                         0, 'bone_original')
        # break
        struct_dict_cropped, interpolation_image = preprocessing.crop_patient_volumes(
            struct_dict,
            slice_axis=slice_axis,
            slice_range=slice_range,
        )

        image, bone_ct = preprocessing.preprocess_bone_image(
            struct_dict_cropped,
            HU_range=HU_range,
        )

        interpolator = preprocessing.get_cached_interpolator(
            output_path_patient,
            interpolation_image,
            voxel_size,
        )

        initial_plane, delta = optimization.parameter_initialization(
            image,
            bone_ct,
            output_path_patient,
            interpolator,
            voxel_size,
            azimuthal_deg_range=azimuthal,
            polar_deg_range=polar,
            initialization_steps=initialization_steps,
        )

        # f = compute_objective_initialization(initial_plane, bone_ct, interpolator, voxel_size)

        # plot_huber_loss([1000, 500, 300], os.path.join(output_path_patient, "huber_loss.pdf"),residual_range=(-3000, 3000), num_points=500)

        optimized_parameters, optimized_objective_values = (
            optimization.run_or_load_optimization(
                output_path_patient,
                image,
                voxel_size,
                delta,
                initial_plane,
                bone_ct,
                interpolator,
                optimized_parameter_list,
                optimized_objective_value_list,
            )
        )

        # optimized_parameters_1, optimized_objective_values_1 = run_or_load_optimization(r"/home/loriskeller/Documents/Master Project/Results/03.08.25/Pure qudratic objective/azimuthal/-20°/4927494",
        #                                                                              image,
        #                                                                              voxel_size,
        #                                                                              delta,
        #                                                                              initial_plane,
        #                                                                              bone_ct,
        #                                                                              interpolator,
        #                                                                              optimized_parameter_list,
        #                                                                              optimized_objective_value_list)

        # robustness_test(
        #     image=image,
        #     bone=bone_ct,
        #     output_path=r"/home/loriskeller/Documents/Master Project/Results/Images for Report/msp results/Robustness/L2",
        #     interpolator_intensity=interpolator,
        #     voxel_size=voxel_size,
        #     azimuthal_deg_range=(0, 45),  # Uses the azimuthal parameter from MSP_pipeline
        #     polar_deg_range=(90, 45),         # Uses the polar parameter from MSP_pipeline
        #     initialization_steps=40,  # Uses the initialization_steps parameter
        #     optimal_plane_params=optimized_parameters   # Uses the optimized parameters as ground truth
        # )
        # break

        # save_3d_slice_as_vector(bone_ct, r"/home/loriskeller/Documents/Master Project/Results/Presentation",
        #                         {'axial': 6, 'coronal': 265, 'sagittal': 285},
        #                         0, 'bone')

        # break
        # assume struct_dict originally has a key "GTVp"
        struct_dict["GTV-T"] = struct_dict.pop("GTVp")
        # plot_and_save_axial_slice(
        #     struct_dict_cropped,
        #     voxel_size,
        #     z_indices=[6],
        #     save_path=os.path.join(r"/home/loriskeller/Documents/Master Project/Results/Images for Report/SASRO", "axial_slice.PNG"),
        #     crop_size=150,
        #     plane_coeffs_list=[optimized_parameters],
        #     optimization_methods_list=["MSP"]
        # )

        # plot_gtvt_positions(
        #     struct_dict_cropped,
        #     voxel_size,
        #     z_index=12,
        #     plane_coeffs=optimized_parameters,
        #     save_path=os.path.join(r"/home/loriskeller/Documents/Master Project/Results/Images for Report", "gtvt_positions.pdf"),
        #     shifts=[25, 10, 0],
        #     crop_size=150,
        #     figsize_per=(5, 5)
        # )

        # optimized_parameters_manual_10587029 = (-0.06283185307179585, 1.6244956418729801, 262)
        # optimized_parameters_manual_10774767 = (-0.0017453292519943233, 1.594660423674561, 240)
        # optimized_parameters_manual_10780163 = (-0.03490658503988659, 1.5982842763770915, 269)

        # plot_axial_coronal(
        #     struct_dict_cropped,
        #     voxel_size,
        #     z_index=6,
        #     y_line=187,
        #     save_path=os.path.join(r"/home/loriskeller/Documents/Master Project/Results/Images for Report", "Extension_diff_1_4mm.pdf"),
        #     plane_coeffs_list=[optimized_parameters],
        #     optimization_methods_list=["MSP optimized"],
        #     crop_size=150,
        #     coronal_crop=(150, 350, 0, 512),
        #     axial_crop=None,
        #     figsize=(12, 6)
        # )
        # display_scrollable_views(struct_dict_cropped, voxel_size)

        # break
        if verification_widget:
            # %matplotlib widget
            msp_widget.display_scrollable_views(
                struct_dict_cropped,
                voxel_size,
                plane_coeffs_list=[optimized_parameters],
                optimization_methods_list=["MSP optimized"],
            )

        if parameter_tweaking:
            # %matplotlib widget
            # Interactive widget with sliders to adjust plane parameters

            visualization.display_interactive_plane_widget(
                struct_dict=struct_dict_cropped,
                voxel_size=voxel_size,
                initial_plane_params=optimized_parameters,
                theta_range=(-10, 10),
                phi_range=(80, 100),
                L_range=(200, 300),
            )

        if patient is not None:
            break

    if patient is None:
        np.save(
            os.path.join(output_path, "optimized_parameters.npy"),
            np.array(optimized_parameter_list),
        )
        np.save(
            os.path.join(output_path, "objective_values.npy"),
            np.array(optimized_objective_value_list),
        )

    end_pipeline = time.time()
    if patient is None:
        print(
            f"Pipeline completed for {len(folders)} patients in {end_pipeline - start_pipeline:.2f} seconds.",
        )
    else:
        print(
            f"Pipeline completed for patient {patient} in {end_pipeline - start_pipeline:.2f} seconds.",
        )
