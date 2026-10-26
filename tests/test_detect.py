import dataclasses

import numpy as np
import pytest
from conftest import WHOLE_SCAN, make_phantom, write_case

from mspdetect import DetectionConfig, Plane, compute_metrics, detect_msp, load_case


def distance_at(plane_a, plane_b, point):
    return abs(plane_a.signed_distance(point) - plane_b.signed_distance(point))


def test_recovers_known_plane(phantom, phantom_result):
    _, true_plane = phantom
    plane = phantom_result.plane
    assert plane.angle_to(true_plane) < 0.5
    assert distance_at(plane, true_plane, np.zeros(3)) < 0.5


def test_single_start_result(phantom_result):
    assert len(phantom_result.starts) == 1
    assert phantom_result.best_start == 0
    assert phantom_result.first_start_agrees
    start = phantom_result.starts[0]
    assert start.initial == phantom_result.grid.planes[0]
    assert start.trajectory[0] == start.initial and start.trajectory[-1] == start.final
    assert len(start.trajectory) == len(start.huber) == len(start.quadratic)
    assert np.all(np.diff(phantom_result.grid.values) >= 0)


def test_multistart_returns_best():
    case, true_plane = make_phantom(yaw_deg=-3.0, roll_deg=2.0, shift_mm=-1.0)
    result = detect_msp(case, DetectionConfig(crop={"reference": None}, optimizer={"n_starts": 3}))
    assert len(result.starts) == 3
    # each start uses the delta of its own grid point
    np.testing.assert_array_equal([s.delta for s in result.starts], result.grid.deltas[:3])
    # the best start is the one with the lowest final quadratic loss
    quadratic = [s.final_quadratic for s in result.starts]
    assert result.best_start == int(np.argmin(quadratic))
    assert result.plane == result.starts[result.best_start].final
    assert result.delta == result.starts[result.best_start].delta
    assert result.plane.angle_to(true_plane) < 0.5
    assert result.first_start_agrees


def test_crop_reference_is_used(phantom):
    case, true_plane = phantom
    result = detect_msp(case, DetectionConfig(crop={"reference": "GTVp", "margins": 4}))
    z = np.flatnonzero(case.masks["GTVp"].any(axis=(0, 1)))
    assert result.crop.start[2] == max(0, z[0] - 4)
    assert result.crop.stop[2] == min(case.image.shape[2], z[-1] + 1 + 4)
    assert result.plane.angle_to(true_plane) < 1.0


@pytest.mark.parametrize("flip", [(0,), (0, 1), (2,)])
def test_laterality_independent_of_storage(tmp_path, phantom, phantom_result, flip):
    """The same physical scan stored in RAS or e.g. LPS gives the same plane and sides."""
    case, _ = phantom
    structures = {"GTVp": "GTVp", "Body": "Body"}
    ras = load_case(write_case(tmp_path / "ras", case), structures=structures)
    flipped = load_case(write_case(tmp_path / "flipped", case, flip=flip), structures=structures)
    np.testing.assert_allclose(flipped.affine, ras.affine, atol=1e-6)

    plane_ras = detect_msp(ras, WHOLE_SCAN).plane
    plane_flipped = detect_msp(flipped, WHOLE_SCAN).plane
    assert plane_flipped.angle_to(plane_ras) < 1e-3
    assert distance_at(plane_flipped, plane_ras, np.zeros(3)) < 1e-3

    m_ras = compute_metrics(ras, plane_ras, structures=["GTVp"])
    m_flipped = compute_metrics(flipped, plane_flipped, structures=["GTVp"])
    assert m_ras["GTVp_side"] == m_flipped["GTVp_side"] == "left"
    assert m_ras["GTVp_left_cc"] == pytest.approx(m_flipped["GTVp_left_cc"])


def test_empty_bone_raises(phantom):
    case, _ = phantom
    with pytest.raises(ValueError, match="no voxels within hu_range"):
        detect_msp(case, DetectionConfig(crop={"reference": None}, hu_range=(5000, 6000)))


def test_select_plane(phantom_result):
    other = Plane([1, 0.1, 0], 3.0)
    result = dataclasses.replace(phantom_result, plane=other)
    assert result.select_plane("best") == other
    assert result.select_plane("first_start") == result.starts[0].final
    with pytest.raises(ValueError, match="Unknown plane"):
        result.select_plane("last")
