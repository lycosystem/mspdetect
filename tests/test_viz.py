import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("ipywidgets")
matplotlib.use("Agg")
pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")

from mspdetect import Plane, save_result  # noqa: E402
from mspdetect.results import load_manual_plane  # noqa: E402
from mspdetect.viz import _plane_line, adjust_plane, adjusted_plane, qc_figure, viewer  # noqa: E402


def test_plane_line_lies_on_plane(phantom, phantom_result):
    case, _ = phantom
    x, y = _plane_line(case, phantom_result.plane, axis=2, index=10)
    indices = np.column_stack([x, y, np.full_like(x, 10)])
    d = phantom_result.plane.signed_distance_voxels(indices, case.affine)
    np.testing.assert_allclose(d, 0, atol=1e-9)


def test_qc_figure(tmp_path, phantom, phantom_result):
    case, true_plane = phantom
    fig = qc_figure(
        case,
        [phantom_result.plane, true_plane],
        labels=["detected", "true"],
        structures=["GTVp"],
    )
    fig.savefig(tmp_path / "qc.png")
    assert (tmp_path / "qc.png").stat().st_size > 0


def test_adjusted_plane():
    plane, pivot = Plane([1, 0, 0], 5.0), np.array([5.0, 10.0, -3.0])
    assert adjusted_plane(plane, pivot) == plane
    shifted = adjusted_plane(plane, pivot, shift_mm=2.0)
    assert shifted.signed_distance(pivot) == pytest.approx(-2.0)  # moved 2 mm to the right
    for kwargs in [{"axial_deg": 10.0}, {"coronal_deg": -7.0}]:
        rotated = adjusted_plane(plane, pivot, **kwargs)
        assert rotated.angle_to(plane) == pytest.approx(abs(next(iter(kwargs.values()))))
        assert rotated.signed_distance(pivot) == pytest.approx(0, abs=1e-9)  # pivot stays
        assert rotated.normal[0] > 0


def test_viewer_runs(phantom, phantom_result):
    case, true_plane = phantom
    viewer(case, [phantom_result.plane, true_plane], labels=["detected", "true"])


def test_adjust_plane_saves_correction(tmp_path, phantom, phantom_result):
    case, _ = phantom
    save_result(phantom_result, tmp_path)
    box = adjust_plane(case, phantom_result, tmp_path)
    axial, _, shift = box.children[0].children
    note, save = box.children[3].children
    status = box.children[4]

    axial.value, shift.value = 3.0, 1.5
    save.click()  # without a note: refused
    assert "Not saved" in status.value
    assert load_manual_plane(tmp_path) is None

    note.value = "dental artefacts"
    save.click()
    manual = load_manual_plane(tmp_path)
    assert manual.note == "dental artefacts"
    assert manual.adjustment == {"axial_deg": 3.0, "coronal_deg": 0.0, "shift_mm": 1.5}
    assert manual.detected_plane == phantom_result.first_start_plane
    assert manual.plane.angle_to(manual.detected_plane) == pytest.approx(3.0, abs=0.01)

    # reopening starts from the saved correction
    box = adjust_plane(case, phantom_result, tmp_path)
    assert box.children[0].children[0].value == 3.0
    assert box.children[3].children[0].value == "dental artefacts"
