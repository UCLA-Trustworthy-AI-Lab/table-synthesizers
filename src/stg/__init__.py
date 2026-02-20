import importlib
import sys

# Backward-compatible alias for environments where the on-disk package is
# `GReaT` but callers import `stg.GREAT`.
try:
    _great_pkg = importlib.import_module(".GReaT", __name__)
    _great_syn_mod = importlib.import_module(".GReaT.great_synthesizer", __name__)
    sys.modules[f"{__name__}.GREAT"] = _great_pkg
    sys.modules[f"{__name__}.GREAT.great_synthesizer"] = _great_syn_mod
except Exception:
    pass

from .tableSynthesizer import TableSynthesizer
from .metrics_manager import (
    MetricsManager,
    MetricsLogger,
    get_metrics_manager,
    create_algorithm_logger,
    TVAELogger,
    CTGANLogger,
    PATECTGANLogger,
    SMOTELogger
)
from . import zero_workaround
