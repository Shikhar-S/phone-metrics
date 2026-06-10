"""phone-metrics: evaluation metrics for phone segmentation and recognition.

Decoupled from the models that produce boundaries/labels. Currently provides
raw boundary-level segmentation scoring (:class:`PrecisionRecallMetric`) plus
ground-truth loaders that read TIMIT and VoxAngeles as distributed.
"""

from .datasets import Utterance, boundary_secs, load_timit, load_voxangeles
from .segmentation import PrecisionRecallMetric

__all__ = [
    "PrecisionRecallMetric",
    "Utterance",
    "boundary_secs",
    "load_timit",
    "load_voxangeles",
]
