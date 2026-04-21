from .base import BaseExperiment
from .apply import ApplyCircuitExperiment
from .stitch import StitchCircuitExperiment
from .ensemble_distill import EnsembleDistillExperiment
from .lth_prune import LTHPruneExperiment
from .cka_compare import CKACompareExperiment

REGISTRY = {
    "apply":  ApplyCircuitExperiment,
    "stitch": StitchCircuitExperiment,
    "ensemble_distill": EnsembleDistillExperiment,
    "lth_prune": LTHPruneExperiment,
    "cka": CKACompareExperiment,
}
