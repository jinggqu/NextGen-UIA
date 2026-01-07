"""
Dataset modules for medical ultrasound image analysis.

Available modules:
- classification: Full dataset for classification tasks
- segmentation: Full dataset for segmentation tasks
- fewshot_classification: Few-shot sampling for classification
- fewshot_segmentation: Few-shot sampling for segmentation
"""

from . import classification
from . import segmentation
from . import fewshot_classification
from . import fewshot_segmentation

__all__ = [
    "classification",
    "segmentation",
    "fewshot_classification",
    "fewshot_segmentation",
]
