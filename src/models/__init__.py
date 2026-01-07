"""
Models module for NextGen-UIA

Provides implementations for various vision-language models and baselines:
- CLIP variants: OpenAI CLIP, BiomedCLIP, MetaCLIP, UnimedCLIP
- Segmentation: CLIPSeg
- Vision models: DINOv2
- Baselines: ResNet, UNet

Each model submodule provides:
- finetune.py: Fine-tuning with Mona/LoRA adapters
- classification.py: Supervised classification
- segmentation.py: Supervised segmentation
- zero_shot.py: Zero-shot classification (for CLIP variants)
- retrieval.py: Image-text retrieval (for BiomedCLIP)
"""

__all__ = [
    "clip",
    "biomedclip",
    "metaclip",
    "unimedclip",
    "clipseg",
    "dino",
    "baselines",
]

