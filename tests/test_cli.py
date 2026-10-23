import csv
import json

from conftest import write_case

from mspdetect.cli import main


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_run_and_metrics(tmp_path, phantom):
    case, _ = phantom
    write_case(tmp_path / "data" / "p1", case)
    (tmp_path / "data" / "p2").mkdir()  # no CT -> recorded as failure
    config = tmp_path / "config.yaml"
    config.write_text(
        "data:\n"
        "  root: data\n"
        "  structures: {GTVp: GTVp, Body: Body}\n"
        "output: results\n"
        "detection:\n"
        "  crop: {reference: GTVp, margins: 5}\n"
        "metrics:\n"
        "  structures: [GTVp, Body]\n"
        "  volume_region: slab\n",
    )
    assert main(["check-config", str(config)]) == 0
    assert main(["run", str(config)]) == 1  # one failure

    results = tmp_path / "results"
    rows = read_csv(results / "msp_results.csv")
    assert [r["case_id"] for r in rows] == ["p1"]
    failures = read_csv(results / "failures.csv")
    assert failures[0]["case_id"] == "p2" and "FileNotFoundError" in failures[0]["error"]
    assert (results / "p1" / "msp.json").exists() and (results / "p1" / "details.npz").exists()

    # second run reuses the existing result
    mtime = (results / "p1" / "msp.json").stat().st_mtime
    main(["run", str(config)])
    assert (results / "p1" / "msp.json").stat().st_mtime == mtime

    assert main(["metrics", str(config)]) == 0
    metrics = read_csv(results / "metrics.csv")
    assert metrics[0]["GTVp_side"] == "left"
    assert metrics[0]["plane"] == "first_start"
    assert (metrics[0]["start"], metrics[0]["is_first_start"]) == ("0", "True")
    assert float(metrics[0]["Body_left_cc"]) > 0
    # volumes are counted within the slices used for the detection
    crop = json.loads((results / "p1" / "msp.json").read_text())["crop"]
    assert metrics[0]["volume_region"] == "slab"
    assert int(metrics[0]["slab_start"]) == crop["start"][2]
    assert int(metrics[0]["slab_stop"]) == crop["stop"][2]

    assert metrics[0]["plane_source"] == "detected"

    # a manual correction replaces the detected plane in the metrics
    from mspdetect import Plane
    from mspdetect.results import save_manual_plane

    detected = Plane.from_dict(json.loads((results / "p1" / "msp.json").read_text())["plane"])
    moved = Plane(detected.normal, detected.offset - 30)  # 30 mm to the left: GTV now right
    save_manual_plane(results / "p1", moved, "test correction", detected)
    assert main(["metrics", str(config)]) == 0
    corrected = read_csv(results / "metrics.csv")
    assert len(corrected) == 1
    assert (corrected[0]["plane_source"], corrected[0]["start"]) == ("manual", "manual")
    assert corrected[0]["GTVp_side"] == "right"
    main(["run", str(config)])
    assert read_csv(results / "msp_results.csv")[0]["manual_correction"] == "True"
    (results / "p1" / "manual.json").unlink()

    # a separate metrics output directory
    config.write_text(config.read_text() + "  output: other_metrics\n")
    assert main(["metrics", str(config)]) == 0
    assert read_csv(tmp_path / "other_metrics" / "metrics.csv") == metrics


def test_metrics_for_all_starts(tmp_path, phantom):
    case, _ = phantom
    write_case(tmp_path / "data" / "p1", case)
    config = tmp_path / "config.yaml"
    config.write_text(
        "data: {root: data, structures: {GTVp: GTVp}}\n"
        "output: results\n"
        "detection: {crop: {reference: null}, optimizer: {n_starts: 3}}\n"
        "metrics: {structures: [GTVp], plane: all_starts}\n",
    )
    assert main(["run", str(config)]) == 0
    assert main(["metrics", str(config)]) == 0
    rows = read_csv(tmp_path / "results" / "metrics.csv")
    assert [r["start"] for r in rows] == ["0", "1", "2"]
    assert [r["is_first_start"] for r in rows] == ["True", "False", "False"]
    best = json.loads((tmp_path / "results" / "p1" / "msp.json").read_text())["best_start"]
    assert [r["is_best"] == "True" for r in rows] == [i == best for i in range(3)]
    assert {r["case_id"] for r in rows} == {"p1"}


def test_changed_settings_are_not_mixed(tmp_path, phantom):
    case, _ = phantom
    write_case(tmp_path / "data" / "p1", case)
    config = tmp_path / "config.yaml"
    base = "data:\n  root: data\noutput: results\ndetection:\n  crop: {reference: null}\n"
    config.write_text(base)
    assert main(["run", str(config)]) == 0
    config.write_text(base + "  hu_range: [400, 2800]\n")
    assert main(["run", str(config)]) == 1
    failures = read_csv(tmp_path / "results" / "failures.csv")
    assert "different detection settings" in failures[0]["error"]


def test_invalid_config(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("data: {root: .}\noutput: out\ndetection: {crop: {margin: 3}}\n")
    assert main(["run", str(config)]) == 2
    assert "Invalid configuration" in capsys.readouterr().err


def test_parallel_run_keeps_case_order(tmp_path, phantom):
    case, _ = phantom
    for name in ["b", "a", "c"]:
        write_case(tmp_path / "data" / name, case)
    config = tmp_path / "config.yaml"
    config.write_text(
        "data:\n  root: data\n  patients: [c, a, b]\noutput: results\nn_jobs: 2\n"
        "detection:\n  crop: {reference: null}\n",
    )
    assert main(["run", str(config)]) == 0
    rows = read_csv(tmp_path / "results" / "msp_results.csv")
    assert [r["case_id"] for r in rows] == ["c", "a", "b"]
