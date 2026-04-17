"""Core data models for the MSP detection workflow.

These models define clear contracts between pipeline stages:
- discovery/loading
- preprocessing
- optimization
- result aggregation
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CasePaths:
    """Filesystem locations for one case."""

    case_id: str
    case_dir: Path
    image_path: Path
    structure_paths: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseData:
    """Loaded image and structure volumes for one case."""

    case_id: str
    image: np.ndarray
    voxel_size: np.ndarray
    structures: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessedCase:
    """Preprocessed volumes and crop context for one case."""

    case_id: str
    image: np.ndarray
    bone_image: np.ndarray
    voxel_size: np.ndarray
    crop_start: int
    crop_end: int
    slice_axis: int
    structures: Mapping[str, np.ndarray] = field(default_factory=dict)
    interpolation_image: np.ndarray | None = None
    interpolation_pad: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MSPResult:
    """Final plane estimate and diagnostics for one case."""

    case_id: str
    plane_params: np.ndarray
    objective_value: float
    success: bool
    runtime_s: float | None = None
    message: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """Minimal run configuration shared across batch and single-case runs."""

    input_root: Path
    output_root: Path
    structure_names: tuple[str, ...] = ("Image",)
    required_structures: tuple[str, ...] = ("Image",)
    reference_structure: str = "GTVp"
    slice_axis: int = 2
    hu_range: tuple[int, int] = (300, 2800)
    crop_mode: str = "margins"
    margins: int | tuple[int, int] | None = None
    fixed_size: int | None = None
    azimuthal: tuple[float, float] = (0.0, 30.0)
    polar: tuple[float, float] = (90.0, 40.0)
    initialization_steps: int = 10


@dataclass(frozen=True)
class BatchSummary:
    """Summary of a batch run across multiple cases."""

    total_cases: int
    processed_cases: int
    failed_cases: int
    results: tuple[MSPResult, ...] = ()
    failures: Mapping[str, str] = field(default_factory=dict)
