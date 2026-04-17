"""Using `pydantic`_, we define configurations for the package.

Most importantly, these configurations are part of the CLIs that the package provides.
but they also help with programmatically validating and constructing various objects.

.. _pydantic: https://docs.pydantic.dev/latest/
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    DirectoryPath
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)
from pydantic_settings.sources import DEFAULT_PATH

class DataConfig(BaseModel):
    """Where to load the CT images from and what structures to extract."""

    source: DirectoryPath = Field(
        "Path to the directory where the CT images are stored."
    )
    patients: str | FilePath = Field(
        defaults=None,
        description=(
            "Either a string specifying an individual patient, or a path "
            "to a txt file containing a list of patients."
        ),
    )
    mapping: dict[str | Literal["Image", "GTVp", "Mandible", "Spinal Cord"]] | None = Field(
        default=None,
        description=(
            "Mapping which maps names of dataset structure names"
            " to standard ones."
        ),
    )

class CropConfigBase(BaseModel):
    enabled: bool = Field(
        defaults=None,
        description="Enable cropping or not."
    )
    axis: int = Field(
        description="Axis to slice along."
    )
    reference_structure: Literal["GTVp", "Mandible", "Body"] = Field(
        description="Reference to use for cropping."
    )
    pad: int = Field(
        description="Padding to add for interpolator."
    )

class ZCropConfig(CropConfigBase):
    mode: Literal["margins", "fixed_size"] = Field(
        default="fixed_size",
        description=(
            "Cropping method to apply. Either adding margins "
            "on top and bottom of the reference structure or "
            "a fixed size window which is symmetrically positioned "
            "around the reference structures."
        ),
    )
    size: int = Field(
        description="Number of voxels of the cropping window."
    )
    margins: int | list[int] = Field(
        description=(
            "Margins to add around the reference structures. "
            "If just an integer it is symmetrically applied, otherwise "
            "the first one is applied below and the other at the top."
        )
    )

class XCropConfig(CropConfigBase):
    size: int = Field(
        description=(
            "Number of voxels to add around center of mass."
        )
    )

