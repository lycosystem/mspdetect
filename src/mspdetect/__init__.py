"""mspdetect: automatic mid-sagittal plane detection in head and neck planning CTs.

Typical use::

    import mspdetect as msp

    case = msp.load_case("data/patient01", structures={"GTVp": "GTVp", "Body": "External"})
    result = msp.detect_msp(case)
    metrics = msp.compute_metrics(case, result.plane, structures=["Parotid_L"])
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mspdetect")
except PackageNotFoundError:  # not installed
    __version__ = "unknown"

from mspdetect.config import (
    BatchConfig,
    CropConfig,
    DataConfig,
    DetectionConfig,
    MetricsConfig,
    OptimizerConfig,
)
from mspdetect.detect import MSPResult, detect_msp
from mspdetect.io import Case, load_case
from mspdetect.metrics import compute_metrics, side_volumes, target_metrics
from mspdetect.plane import Plane
from mspdetect.results import load_result, save_result

__all__ = [
    "BatchConfig",
    "Case",
    "CropConfig",
    "DataConfig",
    "DetectionConfig",
    "MSPResult",
    "MetricsConfig",
    "OptimizerConfig",
    "Plane",
    "__version__",
    "compute_metrics",
    "detect_msp",
    "load_case",
    "load_result",
    "save_result",
    "side_volumes",
    "target_metrics",
]
