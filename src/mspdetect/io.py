"""Module defining function for loading datasets and saving detected MSPs"""

from __future__ import annotations

import gzip
import logging
import os

import nibabel as nib
import numpy as np


def load_nifti_file(file_path):
    """Load a NIfTI file and return its data and voxel size, reoriented to (coronal, sagittal, axial).

    Parameters
    ----------
    file_path : str
        Path to the NIfTI file.

    Returns
    -------
    tuple
        data : np.ndarray
            The image data array, reoriented to (coronal, sagittal, axial).
        voxel_size : np.ndarray
            The voxel size for each axis, reordered to match the data orientation.

    """
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    voxel_size = np.array(nifti_img.header.get_zooms())

    # determine which data-axis is world-X, world-Y, world-Z
    codes = nib.orientations.aff2axcodes(nifti_img.affine)
    code2world = {"L": 0, "R": 0, "P": 1, "A": 1, "I": 2, "S": 2}
    world_of_axis = [code2world[c] for c in codes]
    # find data-axis indices for each world axis
    idx_x = world_of_axis.index(0)  # sagittal
    idx_y = world_of_axis.index(1)  # coronal
    idx_z = world_of_axis.index(2)  # axial

    # we want new_data[ coronal, sagittal, axial ]
    new_order = [idx_y, idx_x, idx_z]
    data = np.transpose(data, new_order)
    voxel_size = np.array(tuple(voxel_size[new_order]))

    return data, voxel_size


def open_gzip_file(gzip_file_path):
    """Open a gzip-compressed file and return its content as bytes.

    Parameters
    ----------
    gzip_file_path : str
        The path to the gzip file.

    Returns
    -------
    bytes or None
        The content of the gzip file, or None if an error occurs.

    """
    try:
        with gzip.open(gzip_file_path, "rb") as f_in:
            file_content = f_in.read()
        return file_content
    except Exception as e:
        print(f"Error opening {gzip_file_path}: {e}")
        return None


def get_image_and_voxel_size_from_gzip(gzip_file_path):
    """Extract the image array and voxel size from a gzipped NIfTI file.

    Parameters
    ----------
    gzip_file_path : str
        Path to the gzipped NIfTI file.

    Returns
    -------
    tuple
        img_data : np.ndarray or None
            The image data array, or None if loading fails.
        voxel_size : np.ndarray or None
            The voxel size for each axis, or None if loading fails.

    """
    file_content = open_gzip_file(gzip_file_path)
    if file_content is not None:
        with open("temp_nifti.nii", "wb") as temp_file:
            temp_file.write(file_content)

        # 🔹 Debugging: Check if the file was actually written
        file_size = os.path.getsize("temp_nifti.nii")
        if file_size == 0:
            print(
                f"❌ Error: 'temp_nifti.nii' was written but is empty! ({gzip_file_path})",
            )
            return None, None

        img_data, voxel_size = load_nifti_file("temp_nifti.nii")
        os.remove("temp_nifti.nii")  # Remove temp file after reading

        return img_data, voxel_size
    print(f"❌ Error: Failed to read file content from '{gzip_file_path}'")
    return None, None


def get_image_and_voxel_size_from_gzip(gzip_file_path):
    """Extract the image array and voxel size from a gzipped NIfTI file.

    Parameters
    ----------
    gzip_file_path : str
        Path to the gzipped NIfTI file.

    Returns
    -------
    tuple
        img_data : np.ndarray or None
            The image data array, or None if loading fails.
        voxel_size : np.ndarray or None
            The voxel size for each axis, or None if loading fails.

    """
    file_content = open_gzip_file(gzip_file_path)
    if file_content is not None:
        with open("temp_nifti.nii", "wb") as temp_file:
            temp_file.write(file_content)

        # 🔹 Debugging: Check if the file was actually written
        file_size = os.path.getsize("temp_nifti.nii")
        if file_size == 0:
            print(
                f"❌ Error: 'temp_nifti.nii' was written but is empty! ({gzip_file_path})",
            )
            return None, None

        img_data, voxel_size = load_nifti_file("temp_nifti.nii")
        os.remove("temp_nifti.nii")  # Remove temp file after reading

        return img_data, voxel_size
    print(f"❌ Error: Failed to read file content from '{gzip_file_path}'")
    return None, None


def load_patient_structures(patient_folder: str, structure_names=None):
    """Load specified structures from a patient folder containing NIfTI or gzipped NIfTI files.

    Parameters
    ----------
    patient_folder : str
        Path to the folder containing patient structure files.
    structure_names : list of str, optional
        List of structure names to load (default: ['Image']).

    Returns
    -------
    dict
        Dictionary mapping structure names to (image, voxel_size) tuples.

    """
    if structure_names is None:
        structure_names = ["Image"]

    struct_dict = {}
    for root, _, files in os.walk(patient_folder):
        for f in files:
            # Only NIfTI files
            if not (f.endswith(".nii.gz") or f.endswith(".nii")):
                continue

            name_part = f

            # Remove extension
            if name_part.endswith(".nii.gz"):
                base = name_part[:-7]
            else:
                base = name_part[:-4]

            # Compare with expected structure names (case-insensitive)
            for struct_name in structure_names:
                if base.lower() == struct_name.lower():
                    file_path = os.path.join(root, f)
                    try:
                        if f.endswith(".nii.gz"):
                            img, voxel_size = get_image_and_voxel_size_from_gzip(
                                file_path,
                            )
                        else:
                            img, voxel_size = load_nifti_file(file_path)
                    except Exception as e:
                        logging.error(
                            f"Error loading '{struct_name}' from {file_path}: {e}",
                        )
                        continue

                    if img is not None:
                        struct_dict[struct_name] = (img, voxel_size)
                        print(f"Loaded '{struct_name}' from {file_path}")
                    break
    return struct_dict
