# mspdetect

Automatic detection of the **mid-sagittal plane (MSP)** in head and neck planning CTs, and
metrics of structures relative to it (left/right volumes, laterality and distance of the
tumour to the midline).

The plane is found by maximising the left-right symmetry of the bone: bone voxels are mirrored
across a candidate plane and compared with the CT intensity at the mirrored position.

> **Research software.** mspdetect is not a medical device and must not be used for clinical
> decisions. Always check results visually (see [Visualization](https://mspdetect.readthedocs.io/en/stable/visualization.html)).

## Installation

Requires Python ≥ 3.10.

```bash
pip install "mspdetect[viz]"
```

`[viz]` adds plotting (matplotlib, ipywidgets) and can be omitted.

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

See the [command line documentation](https://mspdetect.readthedocs.io/en/stable/cli.html) for the behaviour of the commands and all configuration options.

## Inputs and outputs

Input are NIfTI files containing the CT in Hounsfield units and, optionally, binary structure masks of the Body, GTVp, or other structures. All volumes are reoriented to RAS+ on loading.

The outputs contain the detected plane, objective values along other information such as the package version, settings, etc. Planes are stored in **world coordinates**.

When calculating metrics on the detected planes, for each specified structure (`metrics.structures`) a handful of information related to the detected MSP will be determined, e.g. the side, the distance to the MSP, lateral extent, etc.

See the [documentation](https://mspdetect.readthedocs.io/en/stable/conventions.html) for further details.


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


## Citation

If you use mspdetect, please cite the article in [`CITATION.cff`](https://github.com/lycosystem/mspdetect/blob/main/CITATION.cff).
