"""
Adapters module for parameter-efficient fine-tuning

Provides adapter implementations for CLIP and variants:
- mona.py: MONA (Multi-scale Orthogonal Non-linear Adapter) variants
  * BaselineMona, FractionalMona, NoiseAwareMona, FreqEnhancedMona
  * Supports OpenAI CLIP ([N,B,D]) and open_clip models ([B,N,D])
- lora.py: LoRA (Low-Rank Adaptation) implementation
  * Supports OpenAI CLIP and BiomedCLIP
- prompt_tuning.py: Prompt Tuning (CoOp-style)
  * SimplePromptTuner: learns text feature offsets
  * Preserves vision-text alignment by only tuning text side

Supported Models:
- OpenAI CLIP: inject_mona_variant_to_clip, inject_lora_to_clip
- MetaCLIP: inject_mona_variant_to_open_clip
- BiomedCLIP: inject_mona_variant_to_open_clip, inject_lora_to_biomedclip
- UniMedCLIP: inject_mona_variant_to_open_clip
"""

from .mona import (
    BaselineMona,
    FractionalMona,
    NoiseAwareMona,
    FreqEnhancedMona,
    FreqEnhancedMonaOp,
    inject_mona_variant_to_clip,
    inject_mona_variant_to_open_clip,
)

from .lora import (
    inject_lora_to_clip,
    inject_lora_to_biomedclip,
)

from .prompt_tuning import (
    SimplePromptTuner,
    create_simple_prompt_tuner,
)

__all__ = [
    # MONA adapter variants
    "FreqEnhancedMona",
    "FreqEnhancedMonaOp",
    "BaselineMona",
    "FractionalMona",
    "NoiseAwareMona",
    # Injection functions for OpenAI CLIP (sequence-first)
    "inject_mona_variant_to_clip",
    "inject_lora_to_clip",
    # Injection functions for open_clip models (batch-first)
    "inject_mona_variant_to_open_clip",
    "inject_lora_to_biomedclip",
    # Prompt Tuning
    "SimplePromptTuner",
    "create_simple_prompt_tuner",
]
