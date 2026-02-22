from .base import BaseExperiment
from .apply import ApplyCircuitExperiment
from .stitch import StitchCircuitExperiment

REGISTRY = {
    "apply":  ApplyCircuitExperiment,
    "stitch": StitchCircuitExperiment,
}
