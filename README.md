# mspdetect

Automatic detection of the **mid-sagittal plane (MSP)** in head and neck planning CTs, and
metrics of structures relative to it (left/right volumes, laterality and distance of the
tumour to the midline).

The plane is found by maximising the left-right symmetry of the bone: bone voxels are mirrored
across a candidate plane and compared with the CT intensity at the mirrored position.

> **Research software.** mspdetect is not a medical device and must not be used for clinical
> decisions. Always check results visually (see [Quality control](#quality-control)).

## Installation

Requires Python ≥ 3.10.

```bash
pip install "mspdetect[viz] @ git+https://github.com/noemibuehrer/msp-detection-ct"
```

`[viz]` adds plotting (matplotlib, ipywidgets) and can be omitted.

## Input data

One directory per patient with NIfTI files (`.nii` or `.nii.gz`, searched recursively):
the CT in Hounsfield units and, optionally, binary structure masks on the **same voxel grid**
as the CT (checked on loading). Files are identified by name without extension
(case-insensitive), e.g. `CT.nii.gz`, `ROI__[GTVp].nii.gz`.

Only the CT is required. Some structure names have a special meaning:

| Name | Used for |
|---|---|
| `GTVp` (or any name set as `crop.reference`) | selecting the slices used to fit the plane |
| `Body` | masking out the couch and, with `lateral_extent_mm`, centring the shoulder crop |
| `Mandible` | extending the slab if the reference lies just below the mandible |

## Usage

### Python

```python
import mspdetect as msp

case = msp.load_case(
    "data/patient01",
    image="CT",
    structures={"GTVp": "ROI__[GTVp]", "Body": "ROI__[EXTERNAL]", "Parotid_L": "ROI__[Parotid_L]"},
)
result = msp.detect_msp(case)                     # default settings, see DetectionConfig
print(result.plane)                               # normal and offset in world coordinates

metrics = msp.compute_metrics(case, result.plane, structures=["Parotid_L"], target="GTVp")
msp.save_result(result, "results/patient01")      # msp.json + details.npz
```

Settings are passed as a `DetectionConfig`, e.g.
`msp.DetectionConfig(crop={"reference": None})` to use the whole scan.

### Command line

```bash
mspdetect check-config config.yaml   # validate and print all settings (incl. defaults)
mspdetect run config.yaml            # detect the plane of all patients
mspdetect metrics config.yaml        # compute metrics from the detected planes
```

See [`examples/config.example.yaml`](examples/config.example.yaml) for all options. Unknown or
invalid settings are rejected. Patients that fail (e.g. missing CT) are listed in
`failures.csv` and do not stop the run.

`mspdetect run` skips patients that already have a result computed with the **same** detection
settings. If the settings changed, it refuses to mix results: use a new output directory or
`overwrite: true`.

## Conventions

- All volumes are reoriented to RAS+ on loading. Planes are stored in **world coordinates**
  (scanner RAS+ in mm, as given by the NIfTI affine): the set of points `w` with
  `normal · w = offset`.
- The normal points to the patient's **right**. Signed distances are positive on the patient's
  right and negative on the left, independent of how the images were stored.
- Volumes are in cc (= cm³ = ml), distances in mm and angles in degrees. `azimuth_deg`/`polar_deg` are the
  spherical angles of the normal; a plane aligned with the scanner's sagittal plane has
  azimuth 0° and polar angle 90°.

## Outputs

Per patient, in `<output>/<patient>/`:

- `msp.json`: plane, objective value, settings, package version, crop and a summary of each
  optimisation start.
- `details.npz`: grid search (`grid_planes`, `grid_values`, `grid_deltas`) and, per start
  `NN`, the plane at every iteration (`startNN_trajectory`) with the Huber and quadratic
  objective (`startNN_huber`, `startNN_quadratic`). Planes are rows `[nx, ny, nz, offset_mm]`.
  Use `msp.load_result(directory)` to read both files.
- `manual.json` (only after a manual correction, see [Quality control](#quality-control)): the
  corrected plane, the detected plane it replaces, a note, date, user and package version.

Tables in `<output>/`: `msp_results.csv` (one row per patient; `manual_correction` marks
patients with a `manual.json`) and `failures.csv`.
`mspdetect metrics` reads the planes from `<output>/` and writes `metrics.csv` and
`metrics_failures.csv` to `metrics.output` if set (e.g. to compare several metric settings for
the same planes), otherwise to `<output>/`.

## Metrics

For every structure in `metrics.structures`: `<name>_left_cc` and `<name>_right_cc`, the
volume on each side of the plane (counted by voxel centres). `metrics.volume_region` sets
which part of the structures is counted:

- `whole` (default): the whole structures. For long structures such as the spinal cord this
  includes parts far from where the plane was fitted, where neck flexion or rotation moves
  them off the plane.
- `slab`: only the slices used to detect the plane, i.e. the cranial-caudal range of
  `detection.crop` (`fixed_size` or `margins` around the reference, mandible rule,
  `slice_range`), with the full left-right and anterior-posterior extent. The columns
  `slab_start` and `slab_stop` give the slice range `[start, stop)` (canonical orientation,
  counted from inferior). In Python: `compute_metrics(..., slices=result.crop_slices)`.

The target metrics always use the whole target structure. For the `target` structure
(default `GTVp`), prefixed with its name:

| Column | Definition |
|---|---|
| `side` | side with the larger volume (`left`, `right`, `central` if equal) |
| `crosses_midline` | the structure has voxels on both sides |
| `distance_msp_mm` | if crossing: how far the structure extends into the side with the smaller volume (> 0); otherwise minus the distance from the plane to the closest voxel (< 0) |
| `lateral_extent_mm` | extent perpendicular to the plane |
| `left_max_mm`, `right_max_mm` | largest distance from the plane on each side (0 if none) |
| `max_diameter_mm` | largest distance between two voxels of the structure |

With `n_starts > 1`, `metrics.plane` selects the plane: `first_start` (default) uses the
optimisation started from the grid point with the lowest initial value, `best` the lowest
final quadratic loss over all starts, and `all_starts` computes the metrics for every start.
`metrics.csv` has one row per patient and evaluated start ("long" format), with the columns
`start` (index of the start), `is_first_start` and `is_best`. To compare the starts:

```python
df = pd.read_csv("metrics.csv")
df.groupby("case_id")["GTVp_distance_msp_mm"].agg(["min", "max", "std"])  # spread per patient
df[df.is_first_start]                                                     # first start only
```

In Python: `result.starts[i].final` is the plane of start `i`; `result.select_plane("first_start")`
or `result.plane` (= `"best"`).

## Method

1. **Crop** a slab of slices around the reference structure (fixed margins or a fixed-size
   window), optionally extended to the mandible and cropped laterally to exclude the shoulders.
2. **Bone**: voxels inside `Body` with HU in `hu_range` (default 300–2800).
3. **Grid search** over the normal's azimuth and polar angle, for planes through the bone's
   centre of mass, scored with the sum of squared differences between each bone voxel and the
   (cubic-interpolated) CT at its mirror position. A robust Huber threshold δ is estimated
   from the residuals (1.345 · 1.4826 · MAD).
4. **Optimisation** of the Huber loss with BFGS, starting from the best grid plane and using
   the δ of that plane. With `n_starts > 1`, further starts from the next-best grid planes are
   run, each with the δ of its own grid plane. Since these starts minimise different Huber
   losses, the plane with the lowest final quadratic loss (independent of δ) is returned as
   the best, and `first_start_agrees` reports whether the first start reached the same plane
   (within 0.5° and 0.5 mm).

## Quality control

In a Jupyter notebook:

```python
from mspdetect.viz import adjust_plane, qc_figure, view_result, viewer

# axial and coronal views with sliders for one patient of a batch run: first start,
# best start (if different) and manual correction (if any), with structure contours
view_result("patient01", "config.yaml")

viewer(case, [plane_a, plane_b], labels=["a", "b"])        # any case and planes
qc_figure(case, result.plane).savefig("qc.png")            # static figure
```

**Manual correction.** If the detection failed for a patient, the plane can be corrected by hand:

```python
result_dir = "results/patient01"
adjust_plane(case, msp.load_result(result_dir), result_dir)
```

Sliders rotate the plane in the axial and coronal view (about the centre of the detection crop)
and shift it left-right, starting from the detected plane (first start). "Save correction"
requires a note and writes `manual.json`; the detection results are not changed, and deleting
`manual.json` restores the detected plane. Wherever a correction exists, `mspdetect metrics`
uses it instead of the detected plane(s) (one row, `plane_source = manual`, `start = manual`).

## Citation

If you use mspdetect, please cite the article in [`CITATION.cff`](CITATION.cff).

## Development

```bash
uv pip install -e ".[dev,viz]"
pre-commit install
pytest
```

The tests use synthetic phantoms with a known symmetry plane, including copies stored in
different orientations to check that left and right are never swapped.

## License

MIT, see [LICENSE](LICENSE).
