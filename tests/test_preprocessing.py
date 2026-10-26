import numpy as np
import pytest

from mspdetect import Case, CropConfig
from mspdetect.preprocessing import bone_image, crop_box, lateral_bounds, slice_bounds


def make_case(gtv_slices=(20, 25), mandible_slices=None, body_x=None, n=(40, 30, 60)):
    image = np.zeros(n, dtype=np.float32)
    masks = {"GTVp": np.zeros(n, bool)}
    masks["GTVp"][18:22, 10:15, gtv_slices[0] : gtv_slices[1]] = True
    if mandible_slices is not None:
        masks["Mandible"] = np.zeros(n, bool)
        masks["Mandible"][10:30, 5:10, mandible_slices[0] : mandible_slices[1]] = True
    if body_x is not None:
        masks["Body"] = np.zeros(n, bool)
        masks["Body"][body_x[0] : body_x[1]] = True
    return Case("c", image, np.eye(4), masks)


def test_margins():
    case = make_case()
    assert slice_bounds(case, CropConfig(margins=3)) == (17, 28)
    assert slice_bounds(case, CropConfig(margins=(2, 5))) == (18, 30)


def test_margins_are_clamped():
    case = make_case(gtv_slices=(1, 58))
    assert slice_bounds(case, CropConfig(margins=10)) == (0, 60)


def test_fixed_size():
    case = make_case()
    # extent 20..24, centre 22 -> 22 +- 4
    assert slice_bounds(case, CropConfig(mode="fixed_size", fixed_size=4)) == (18, 27)
    # reference larger than the window: keep the reference
    assert slice_bounds(case, CropConfig(mode="fixed_size", fixed_size=1)) == (20, 25)


def test_mandible_included_when_reference_is_below():
    case = make_case(gtv_slices=(20, 25), mandible_slices=(27, 40))
    config = CropConfig(margins=0)
    assert slice_bounds(case, config) == (20, 33)
    assert slice_bounds(case, config.model_copy(update={"include_mandible": False})) == (20, 25)


def test_mandible_ignored_when_reference_is_above():
    case = make_case(gtv_slices=(40, 45), mandible_slices=(20, 30))
    assert slice_bounds(case, CropConfig(margins=0)) == (40, 45)


def test_slice_range_and_whole_scan():
    case = make_case()
    assert slice_bounds(case, CropConfig(slice_range=(5, 15))) == (5, 15)
    assert slice_bounds(case, CropConfig(reference=None)) == (0, 60)


def test_missing_reference_raises():
    case = make_case()
    with pytest.raises(ValueError, match="crop reference 'Larynx' not loaded"):
        slice_bounds(case, CropConfig(reference="Larynx"))


def test_lateral_crop_centred_on_body():
    case = make_case(body_x=(10, 31))
    assert lateral_bounds(case, CropConfig()) == (0, 40)
    assert lateral_bounds(case, CropConfig(lateral_extent_mm=10)) == (15, 25)
    box = crop_box(case, CropConfig(lateral_extent_mm=10, margins=0))
    assert box.start == (15, 0, 20) and box.stop == (25, 30, 25)


def test_bone_image():
    image = np.array([[[-1000.0, 200.0, 500.0, 3000.0]]])
    np.testing.assert_array_equal(bone_image(image, (300, 2800)), [[[0, 0, 500, 0]]])
    body = np.array([[[True, True, False, True]]])
    np.testing.assert_array_equal(bone_image(image, (300, 2800), body), [[[0, 0, 0, 0]]])
