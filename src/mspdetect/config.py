"""Configuration of MSP detection, metrics and batch processing.

All configurations are validated with `pydantic`_: unknown keys, wrong types and
inconsistent values raise an error instead of being silently ignored.

.. _pydantic: https://docs.pydantic.dev/latest/
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CropConfig(_Config):
    """Which part of the CT is used to fit the plane.

    Slices are counted along the patient's inferior→superior axis. Structures named
    ``Mandible`` and ``Body`` have a special meaning (see ``include_mandible`` and
    ``lateral_extent_mm``; ``Body`` is also used to mask out the couch).
    """

    reference: str | None = Field(
        default="GTVp",
        description=(
            "Structure around which the slab of slices is cropped. "
            "None uses the whole scan (or `slice_range`)."
        ),
    )
    slice_range: tuple[NonNegativeInt, NonNegativeInt] | None = Field(
        default=None,
        description="Explicit [start, stop) slice range; overrides `reference`.",
    )
    mode: Literal["margins", "fixed_size"] = Field(
        default="margins",
        description=(
            "`margins`: extent of the reference plus `margins` slices. "
            "`fixed_size`: 2*`fixed_size` + 1 slices centred on the reference "
            "(or the reference extent if it is larger)."
        ),
    )
    margins: NonNegativeInt | tuple[NonNegativeInt, NonNegativeInt] = Field(
        default=10,
        description="Slices added below/above the reference: one value or [inferior, superior].",
    )
    fixed_size: PositiveInt = Field(
        default=20,
        description="Half-size (slices) of the window used in `fixed_size` mode.",
    )
    include_mandible: bool = Field(
        default=True,
        description=(
            "If the reference ends below or within 5 slices of the lower edge of the "
            "Mandible, extend the slab to include those mandible slices."
        ),
    )
    lateral_extent_mm: float | None = Field(
        default=None,
        gt=0,
        description=(
            "If set, keep only this width (mm, left-right) centred on the Body "
            "(or image centre) to exclude the shoulders."
        ),
    )
    pad: NonNegativeInt = Field(
        default=10,
        description="Extra voxels around each crop used only for interpolating mirrored points.",
    )

    @model_validator(mode="after")
    def _check_slice_range(self):
        if self.slice_range is not None and self.slice_range[0] >= self.slice_range[1]:
            raise ValueError(
                f"slice_range must be [start, stop) with start < stop, got {self.slice_range}"
            )
        return self


class OptimizerConfig(_Config):
    """Grid search initialisation and optimisation of the plane."""

    azimuth_deg: tuple[float, float] = Field(
        default=(0.0, 10.0),
        description="Grid search over the azimuthal angle: [centre, half-width] in degrees.",
    )
    polar_deg: tuple[float, float] = Field(
        default=(90.0, 10.0),
        description="Grid search over the polar angle: [centre, half-width] in degrees.",
    )
    grid_steps: PositiveInt = Field(
        default=10,
        description="Number of angles per axis; the grid has grid_steps**2 planes.",
    )
    n_starts: PositiveInt = Field(
        default=1,
        description=(
            "Number of optimisations, started from the best grid points. 1 uses only "
            "the grid point with the lowest initial value; >1 returns the start with "
            "the lowest final quadratic loss and reports whether the first start agrees with it."
        ),
    )

    @model_validator(mode="after")
    def _check_n_starts(self):
        if self.n_starts > self.grid_steps**2:
            raise ValueError(
                f"n_starts ({self.n_starts}) cannot exceed the number of grid points "
                f"({self.grid_steps**2})",
            )
        if self.azimuth_deg[1] < 0 or self.polar_deg[1] < 0:
            raise ValueError("Half-widths of the angle ranges must be >= 0")
        return self


class DetectionConfig(_Config):
    """Everything that determines the detected plane of one case."""

    hu_range: tuple[float, float] = Field(
        default=(300.0, 2800.0),
        description="Voxels with HU in [low, high] are considered bone.",
    )
    crop: CropConfig = CropConfig()
    optimizer: OptimizerConfig = OptimizerConfig()

    @model_validator(mode="after")
    def _check_hu_range(self):
        if self.hu_range[0] >= self.hu_range[1]:
            raise ValueError(f"hu_range must be [low, high] with low < high, got {self.hu_range}")
        return self

    def fingerprint(self) -> str:
        """Short hash identifying these settings (used to detect stale results)."""
        dump = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(dump.encode()).hexdigest()[:16]


class DataConfig(_Config):
    """Where the cases are and how their files are named."""

    root: Path = Field(description="Directory containing one sub-directory per case.")
    patients: list[str] | Path | None = Field(
        default=None,
        description=(
            "Cases to process: a list of sub-directory names (may contain '/'), a "
            "text file with one name per line, or None for all sub-directories."
        ),
    )
    image: str | list[str] = Field(
        default="CT",
        description="File name(s) of the CT without extension (case-insensitive).",
    )
    structures: dict[str, str | list[str]] = Field(
        default_factory=dict,
        description="File name(s) of structures without extension (case-insensitive).",
    )

    @field_validator("patients", mode="before")
    @classmethod
    def _ids_as_strings(cls, value):
        # YAML reads numeric patient IDs as integers
        if isinstance(value, list):
            return [str(v) for v in value]
        return value


class MetricsConfig(_Config):
    """Which metrics to compute relative to the detected plane."""

    output: Path | None = Field(
        default=None,
        description=(
            "Directory for metrics.csv and metrics_failures.csv; None uses the "
            "top-level `output`. The detected planes are always read from `output`."
        ),
    )
    structures: list[str] = Field(
        default_factory=list,
        description="Structures for which left/right volumes are computed.",
    )
    target: str | None = Field(
        default="GTVp",
        description="Structure for which laterality and distances to the plane are computed.",
    )
    plane: Literal["first_start", "best", "all_starts"] = Field(
        default="first_start",
        description=(
            "Plane(s) used for the metrics: `first_start` = result of the optimisation "
            "started from the grid point with the lowest initial value; `best` = lowest "
            "final quadratic loss over all starts; `all_starts` = one row per start. "
            "Identical if n_starts is 1."
        ),
    )
    volume_region: Literal["whole", "slab"] = Field(
        default="whole",
        description=(
            "Region for left/right volumes: `whole` = the whole structures; `slab` = "
            "only the slices used to detect the plane (full left-right and "
            "anterior-posterior extent)."
        ),
    )


class BatchConfig(_Config):
    """Configuration file of the command line interface."""

    data: DataConfig
    output: Path = Field(description="Directory where results are written.")
    detection: DetectionConfig = DetectionConfig()
    metrics: MetricsConfig = MetricsConfig()
    n_jobs: PositiveInt = Field(default=1, description="Number of cases processed in parallel.")
    overwrite: bool = Field(default=False, description="Recompute cases that already have results.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> BatchConfig:
        """Load a YAML file; relative paths are resolved relative to the file."""
        path = Path(path)
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
        base = path.resolve().parent

        def resolve(value):
            return str(base / value) if isinstance(value, str) else value

        data = raw.get("data")
        if isinstance(data, dict):
            data["root"] = resolve(data.get("root"))
            if isinstance(data.get("patients"), str):
                data["patients"] = resolve(data["patients"])
        if "output" in raw:
            raw["output"] = resolve(raw["output"])
        metrics = raw.get("metrics")
        if isinstance(metrics, dict) and metrics.get("output") is not None:
            metrics["output"] = resolve(metrics["output"])
        return cls.model_validate(raw)

    @property
    def metrics_output(self) -> Path:
        """Directory for the metrics tables: ``metrics.output`` or ``output``."""
        return self.metrics.output or self.output
