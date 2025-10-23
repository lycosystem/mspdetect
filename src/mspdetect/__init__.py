"""mspdetect
=========

A Python package for detecting the mid-sagittal plane (MSP) in CT images.

This package provides utilities for:
- Reading and writing medical image data
- Preprocessing and interpolation
- Geometric transformations
- MSP optimization and detection
- Visualization and batch processing pipelines
"""

from . import (
    geometry,
    io,
    msp_widget,
    optimization,
    pipeline,
    preprocessing,
    visualization,
)

__all__ = [
    "io",
    "preprocessing",
    "geometry",
    "optimization",
    "visualization",
    "pipeline",
    "run_pipeline",
    "msp_widget",
    "DEFAULT_PARAMS",
]
