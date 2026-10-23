import numpy as np
import pytest
from pydantic import ValidationError

from mspdetect import (
    BatchConfig,
    DetectionConfig,
    MetricsConfig,
    OptimizerConfig,
    Plane,
    load_result,
    save_result,
)


def test_unknown_key_rejected():
    with pytest.raises(ValidationError, match="extra"):
        DetectionConfig(crop={"margin": 10})


@pytest.mark.parametrize(
    "kwargs",
    [{"n_starts": 101}, {"grid_steps": 0}, {"azimuth_deg": (0, -5)}],
)
def test_invalid_optimizer(kwargs):
    with pytest.raises(ValidationError):
        OptimizerConfig(**kwargs)


def test_metrics_plane_choice():
    assert MetricsConfig().plane == "first_start"
    assert MetricsConfig(plane="all_starts").plane == "all_starts"
    with pytest.raises(ValidationError):
        MetricsConfig(plane="last")


def test_invalid_ranges():
    with pytest.raises(ValidationError, match="hu_range"):
        DetectionConfig(hu_range=(2800, 300))
    with pytest.raises(ValidationError, match="slice_range"):
        DetectionConfig(crop={"slice_range": (10, 5)})


def test_fingerprint_changes_with_settings():
    assert DetectionConfig().fingerprint() == DetectionConfig().fingerprint()
    assert DetectionConfig().fingerprint() != DetectionConfig(crop={"margins": 5}).fingerprint()


def test_yaml_relative_paths(tmp_path):
    (tmp_path / "data").mkdir()
    config_file = tmp_path / "config.yaml"
    config_file.write_text("data:\n  root: data\n  patients: [101338, abc]\noutput: out\n")
    config = BatchConfig.from_yaml(config_file)
    assert config.data.root == tmp_path / "data"
    assert config.output == tmp_path / "out"
    assert config.data.patients == ["101338", "abc"]
    assert config.metrics_output == tmp_path / "out"  # defaults to `output`

    config_file.write_text("data: {root: data}\noutput: out\nmetrics: {output: metrics/run1}\n")
    config = BatchConfig.from_yaml(config_file)
    assert config.metrics_output == tmp_path / "metrics" / "run1"


def test_result_round_trip(tmp_path, phantom_result):
    save_result(phantom_result, tmp_path)
    loaded = load_result(tmp_path)
    assert loaded.plane == phantom_result.plane
    assert loaded.config == phantom_result.config
    assert loaded.crop == phantom_result.crop
    np.testing.assert_array_equal(loaded.grid.values, phantom_result.grid.values)
    start, original = loaded.starts[0], phantom_result.starts[0]
    assert start.trajectory == original.trajectory
    np.testing.assert_array_equal(start.huber, original.huber)
    assert start.delta == original.delta
    assert start.final_quadratic == original.final_quadratic
    assert not list(tmp_path.glob(".*tmp*"))


def test_manual_plane(tmp_path, phantom_result):
    from mspdetect.results import load_manual_plane, save_manual_plane

    corrected = Plane([1, 0.05, 0], 3.0)
    detected = phantom_result.plane
    with pytest.raises(ValueError, match="No detected result"):
        save_manual_plane(tmp_path, corrected, "note", detected)
    save_result(phantom_result, tmp_path)
    with pytest.raises(ValueError, match="note"):
        save_manual_plane(tmp_path, corrected, "  ", detected)
    assert load_manual_plane(tmp_path) is None

    save_manual_plane(tmp_path, corrected, " tilted by artefacts ", detected, {"shift_mm": 1})
    manual = load_manual_plane(tmp_path)
    assert manual.plane == corrected and manual.detected_plane == detected
    assert manual.note == "tilted by artefacts"
    assert manual.adjustment == {"shift_mm": 1}
    assert manual.user and manual.created
    assert load_result(tmp_path).plane == detected  # detection result unchanged
