import numpy as np
import pytest
from conftest import GTV_DISTANCE, GTV_RADIUS
from pydantic import ValidationError

from mspdetect import Case, MetricsConfig, Plane, compute_metrics, side_volumes, target_metrics

AFFINE = np.diag([2.0, 1.0, 1.0, 1.0])  # 2 mm³ voxels, x = 2 * index
MIDLINE = Plane([1, 0, 0], 21.0)  # x = 21 mm, between voxel 10 (20 mm) and 11 (22 mm)


def case_with(x_range, name="GTVp"):
    mask = np.zeros((20, 10, 10), bool)
    mask[x_range[0] : x_range[1], 2:7, 3:5] = True  # 10 voxels per x index
    return Case("c", np.zeros(mask.shape, np.float32), AFFINE, {name: mask})


def test_side_volumes():
    case = case_with((8, 14))  # x indices 8, 9, 10 left; 11, 12, 13 right
    left, right = side_volumes(case, MIDLINE, "GTVp")
    assert left == pytest.approx(3 * 10 * 2 / 1000)
    assert right == pytest.approx(3 * 10 * 2 / 1000)


def test_target_crossing_midline():
    case = case_with((9, 15))  # left: 9, 10 (max 3 mm); right: 11..14 (max 7 mm)
    m = target_metrics(case, MIDLINE)
    assert m["side"] == "right"
    assert m["crosses_midline"]
    assert m["distance_msp_mm"] == pytest.approx(3.0)
    assert m["lateral_extent_mm"] == pytest.approx(10.0)
    assert (m["left_max_mm"], m["right_max_mm"]) == pytest.approx((3.0, 7.0))


def test_target_not_crossing():
    case = case_with((2, 6))  # all left; closest voxel 5 at 10 mm -> 11 mm from plane
    m = target_metrics(case, MIDLINE)
    assert m["side"] == "left"
    assert not m["crosses_midline"]
    assert m["distance_msp_mm"] == pytest.approx(-11.0)
    assert m["lateral_extent_mm"] == pytest.approx(6.0)


def test_max_diameter():
    case = case_with((2, 6))
    # corners (2,2,3) and (5,6,4) in mm: (4,2,3) and (10,6,4)
    assert target_metrics(case, MIDLINE)["max_diameter_mm"] == pytest.approx(np.sqrt(36 + 16 + 1))


def test_compute_metrics_missing_structure():
    case = case_with((2, 6))
    row = compute_metrics(case, MIDLINE, structures=["GTVp", "Parotid_L"])
    assert np.isnan(row["Parotid_L_left_cc"])
    assert row["GTVp_side"] == "left"
    with pytest.raises(ValueError, match="target structure 'GTVp' not loaded"):
        compute_metrics(case_with((2, 6), name="X"), MIDLINE)


def case_with_cord():
    """GTV in slices 3-4; a 'cord' along all 10 slices, left in 0-4, right in 5-9."""
    shape = (20, 10, 10)
    masks = {"GTVp": np.zeros(shape, bool), "Cord": np.zeros(shape, bool)}
    masks["GTVp"][2:6, 2:7, 3:5] = True
    masks["Cord"][9, 4, 0:5] = True  # x index 9 -> 18 mm: left of the plane at 21 mm
    masks["Cord"][11, 4, 5:10] = True  # x index 11 -> 22 mm: right
    return Case("c", np.zeros(shape, np.float32), AFFINE, masks)


def test_volumes_within_slices():
    case = case_with_cord()
    voxel_cc = 2 / 1000
    slab = compute_metrics(case, MIDLINE, structures=["Cord"], slices=(2, 7))
    assert (slab["slab_start"], slab["slab_stop"]) == (2, 7)
    # slices 2, 3, 4 left; 5, 6 right
    volumes = (slab["Cord_left_cc"], slab["Cord_right_cc"])
    assert volumes == pytest.approx((3 * voxel_cc, 2 * voxel_cc))

    whole = compute_metrics(case, MIDLINE, structures=["Cord"])
    assert "slab_start" not in whole
    volumes = (whole["Cord_left_cc"], whole["Cord_right_cc"])
    assert volumes == pytest.approx((5 * voxel_cc, 5 * voxel_cc))
    # target metrics always use the whole target
    assert whole["GTVp_left_max_mm"] == slab["GTVp_left_max_mm"]


def test_metrics_config_volume_region():
    assert MetricsConfig().volume_region == "whole"
    with pytest.raises(ValidationError):
        MetricsConfig(volume_region="crop")


def test_phantom_gtv(phantom, phantom_result):
    case, _ = phantom
    m = target_metrics(case, phantom_result.plane)
    assert m["side"] == "left" and not m["crosses_midline"]
    voxel = max(case.voxel_size)
    assert m["distance_msp_mm"] == pytest.approx(GTV_DISTANCE + GTV_RADIUS, abs=voxel)
