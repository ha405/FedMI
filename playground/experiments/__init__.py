from .base import BaseExperiment
from .apply import ApplyCircuitExperiment
from .cka_compare import CKACompareExperiment

REGISTRY = {
    "apply":  ApplyCircuitExperiment,
    "cka": CKACompareExperiment,
}
