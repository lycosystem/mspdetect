import numpy as np
import pytest

from mspdetect.plane import Plane, angles_from_normal, array_frame, normal_from_angles


def test_normal_points_to_patient_right():
    plane = Plane([-2.0, 0.0, 0.0], -10.0)
    np.testing.assert_allclose(plane.normal, [1, 0, 0])
    assert plane.offset == pytest.approx(5.0)
    assert plane.signed_distance([6.0, 0, 0]) > 0  # right of the plane
    assert plane.signed_distance([4.0, 0, 0]) < 0


def test_invalid_normal():
    with pytest.raises(ValueError, match="Invalid plane normal"):
        Plane([0, 0, 0], 1.0)


def test_angles_round_trip():
    theta, phi = 0.3, 1.4
    assert angles_from_normal(normal_from_angles(theta, phi)) == pytest.approx((theta, phi))


def test_frame_round_trip():
    rng = np.random.default_rng(0)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    t = rng.normal(size=3) * 50
    plane = Plane(rng.normal(size=3), 12.3)
    n, offset = plane.to_frame(R, t)
    back = Plane.from_frame(n, offset, R, t)
    np.testing.assert_allclose(back.as_array(), plane.as_array(), atol=1e-12)
    # a point on the plane in frame coordinates lies on the plane in world coordinates
    p = n * offset
    assert plane.signed_distance(R @ p + t) == pytest.approx(0, abs=1e-9)


def test_array_frame_matches_affine():
    affine = np.diag([-0.8, -0.8, 2.5, 1.0])
    affine[:3, 3] = [100, 80, -40]
    start = (10, 5, 3)
    R, t = array_frame(affine, start)
    index = np.array([14.0, 9.0, 7.0])
    p = (index - start) * [0.8, 0.8, 2.5]
    np.testing.assert_allclose(R @ p + t, affine[:3, :3] @ index + affine[:3, 3])


def test_signed_distance_voxels():
    affine = np.diag([2.0, 1.0, 3.0, 1.0])
    plane = Plane([1, 0, 0], 5.0)
    d = plane.signed_distance_voxels([[0, 0, 0], [3, 7, 2]], affine)
    np.testing.assert_allclose(d, [-5.0, 1.0])
