"""
MONA Variants for CLIP Fine-tuning

Implements different variants of MONA adapter:
1. Baseline MONA - Multi-scale convolution
2. Noise-aware MONA - Adaptive attention mechanism
3. Frequency-enhanced MONA - Learnable frequency filtering
4. Hybrid MONA - Combines noise-aware attention + frequency filtering

Vision Encoder Only:
MONA is specifically designed for visual features with 2D spatial structure.
It uses multi-scale 2D convolutions which are not suitable for 1D text sequences.
Therefore, MONA is only injected into the vision encoder, NOT the text encoder.

Tensor Format Support:
All MONA variants internally work with [B, N, D] format (batch-first).
- OpenAI CLIP: [N, B, D] (sequence-first) - MONA handles conversion internally
- MetaCLIP: [B, N, D] (batch-first, open_clip) - uses BatchFirstMonaWrapper
- BiomedCLIP: [B, N, D] (batch-first, open_clip) - uses BatchFirstMonaWrapper
- UniMedCLIP: [B, N, D] (batch-first, open_clip) - uses BatchFirstMonaWrapper

Injection Functions:
- inject_mona_variant_to_clip: For OpenAI CLIP (vision encoder only)
- inject_mona_variant_to_open_clip: For open_clip models (vision encoder only)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Batch-First Wrapper for open_clip Models
# ============================================================================


class BatchFirstMonaWrapper(nn.Module):
    """
    Wrapper to adapt MONA (which expects [N, B, D]) to batch-first format ([B, N, D])

    Used by open_clip-based models:
    - MetaCLIP
    - BiomedCLIP
    - UniMedCLIP

    OpenAI CLIP does NOT need this wrapper as it uses [N, B, D] natively.
    """

    def __init__(self, mona_adapter):
        super().__init__()
        self.clip_mona = mona_adapter

    def forward(self, x, hw_shapes=None):
        """
        Args:
            x: (B, N, D) - Batch-first format (open_clip standard)
            hw_shapes: (H, W) spatial dimensions
        Returns:
            output: (B, N, D) - Batch-first format
        """
        # Convert [B, N, D] -> [N, B, D] for MONA
        x = x.permute(1, 0, 2)
        # Apply MONA (internally processes as [B, N, D], returns [N, B, D])
        x = self.clip_mona(x, hw_shapes)
        # Convert back [N, B, D] -> [B, N, D]
        return x.permute(1, 0, 2)


# ============================================================================
# Baseline MONA Variant
# ============================================================================


class BaselineMonaOp(nn.Module):
    """Baseline MONA operation with multi-scale convolution"""

    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        identity = x
        x = self.projector(x)
        return identity + x


class BaselineMona(nn.Module):
    """
    Baseline MONA adapter for CLIP

    Internally works with [B, N, D] format (batch-first).
    Accepts [N, B, D] input from OpenAI CLIP and converts it internally.
    """

    def __init__(self, in_dim, bottleneck_dim=64):
        super().__init__()
        self.project1 = nn.Linear(in_dim, bottleneck_dim)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(bottleneck_dim, in_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.adapter_conv = BaselineMonaOp(bottleneck_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        """
        Args:
            x: (N, B, D) - Sequence-first format from OpenAI CLIP
            hw_shapes: (H, W) spatial dimensions
        Returns:
            output: (N, B, D) - Sequence-first format
        """
        x = x.permute(1, 0, 2)  # [N, B, D] -> [B, N, D]
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax

        x = self.project1(x)

        b, n, c = x.shape
        if hw_shapes is not None:
            h, w = hw_shapes
            cls_token = x[:, 0:1, :]
            spatial_tokens = x[:, 1:, :]

            spatial_tokens = spatial_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)
            spatial_tokens = self.adapter_conv(spatial_tokens)
            spatial_tokens = spatial_tokens.permute(0, 2, 3, 1).reshape(b, h * w, c)

            x = torch.cat([cls_token, spatial_tokens], dim=1)
        else:
            h = w = int(np.sqrt(n))
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
            x = self.adapter_conv(x)
            x = x.permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.nonlinear(x)
        x = self.dropout(x)
        x = self.project2(x)

        result = identity + x
        return result.permute(1, 0, 2)  # [B, N, D] -> [N, B, D]


# ============================================================================
# Noise-aware MONA Variant
# ============================================================================


class NoiseAwareMonaOp(nn.Module):
    """MONA with noise-aware adaptive attention"""

    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

        # Noise estimation and attention
        self.noise_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, in_features // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 4, 3, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        identity = x

        # Compute multi-scale features
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        # Adaptive weighting based on noise level
        weights = self.noise_estimator(x)  # (B, 3, 1, 1)
        w1 = weights[:, 0:1, :, :]
        w2 = weights[:, 1:2, :, :]
        w3 = weights[:, 2:3, :, :]

        x = conv1_x * w1 + conv2_x * w2 + conv3_x * w3 + identity
        identity = x
        x = self.projector(x)
        return identity + x


class NoiseAwareMona(nn.Module):
    """
    Noise-aware MONA adapter for CLIP

    Internally works with [B, N, D] format (batch-first).
    Accepts [N, B, D] input from OpenAI CLIP and converts it internally.
    """

    def __init__(self, in_dim, bottleneck_dim=64):
        super().__init__()
        self.project1 = nn.Linear(in_dim, bottleneck_dim)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(bottleneck_dim, in_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.adapter_conv = NoiseAwareMonaOp(bottleneck_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        """
        Args:
            x: (N, B, D) - Sequence-first format from OpenAI CLIP
            hw_shapes: (H, W) spatial dimensions
        Returns:
            output: (N, B, D) - Sequence-first format
        """
        x = x.permute(1, 0, 2)  # [N, B, D] -> [B, N, D]
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax

        x = self.project1(x)

        b, n, c = x.shape
        if hw_shapes is not None:
            h, w = hw_shapes
            cls_token = x[:, 0:1, :]
            spatial_tokens = x[:, 1:, :]

            spatial_tokens = spatial_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)
            spatial_tokens = self.adapter_conv(spatial_tokens)
            spatial_tokens = spatial_tokens.permute(0, 2, 3, 1).reshape(b, h * w, c)

            x = torch.cat([cls_token, spatial_tokens], dim=1)
        else:
            h = w = int(np.sqrt(n))
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
            x = self.adapter_conv(x)
            x = x.permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.nonlinear(x)
        x = self.dropout(x)
        x = self.project2(x)

        result = identity + x
        return result.permute(1, 0, 2)


# ============================================================================
# Frequency-enhanced MONA Variant
# ============================================================================


class FreqEnhancedMonaOp(nn.Module):
    """
    Frequency-Enhanced MONA operation

    Suppresses periodic artifacts in frequency domain while maintaining
    multi-scale spatial feature extraction capabilities.
    """

    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

        # Learnable frequency filter
        self.freq_filter = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        identity = x

        # Frequency domain filtering
        b, c, h, w = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        x_freq = x_freq * self.freq_filter.view(1, -1, 1, 1)
        x = torch.fft.irfft2(x_freq, s=(h, w), dim=(-2, -1))

        # Multi-scale convolution
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        identity = x
        x = self.projector(x)
        return identity + x


class FreqEnhancedMona(nn.Module):
    """
    Frequency-Enhanced MONA adapter

    Internally works with [B, N, D] format (batch-first).
    Accepts [N, B, D] input from OpenAI CLIP and converts it internally.
    Combines frequency domain filtering with spatial multi-scale processing
    for enhanced feature extraction in medical images.
    """

    def __init__(self, in_dim, bottleneck_dim=64):
        super().__init__()
        self.project1 = nn.Linear(in_dim, bottleneck_dim)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(bottleneck_dim, in_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.adapter_conv = FreqEnhancedMonaOp(bottleneck_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        """
        Args:
            x: (N, B, D) where N = H*W + 1 (with CLS token) - sequence-first format
            hw_shapes: (H, W) spatial dimensions
        """
        # Convert to [B, N, D] for processing
        x = x.permute(1, 0, 2)  # [N, B, D] -> [B, N, D]
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax

        # Project to bottleneck dimension
        x = self.project1(x)  # (B, N, bottleneck_dim)

        # Apply spatial convolution (excluding CLS token if present)
        b, n, c = x.shape
        if hw_shapes is not None:
            h, w = hw_shapes
            # Assume first token is CLS token
            cls_token = x[:, 0:1, :]  # (B, 1, C)
            spatial_tokens = x[:, 1:, :]  # (B, H*W, C)

            # Reshape to 2D spatial format
            spatial_tokens = spatial_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)  # (B, C, H, W)
            spatial_tokens = self.adapter_conv(spatial_tokens)
            spatial_tokens = spatial_tokens.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, H*W, C)

            # Concatenate CLS token back
            x = torch.cat([cls_token, spatial_tokens], dim=1)  # (B, N, C)
        else:
            # If no spatial info, treat as sequence
            h = w = int(np.sqrt(n))
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
            x = self.adapter_conv(x)
            x = x.permute(0, 2, 3, 1).reshape(b, n, c)

        # Apply nonlinearity and project back
        x = self.nonlinear(x)
        x = self.dropout(x)
        x = self.project2(x)  # (B, N, in_dim)

        result = identity + x
        # Convert back to [N, B, D] format
        return result.permute(1, 0, 2)  # [B, N, D] -> [N, B, D]


# ============================================================================
# Hybrid MONA Variant (Noise-aware + Frequency-enhanced)
# ============================================================================


class HybridNoiseFreqMonaOp(nn.Module):
    """
    Hybrid MONA combining noise-aware attention and frequency-domain filtering

    This variant combines:
    1. Frequency-domain filtering for global periodic artifacts
    2. Noise-aware adaptive attention for local noise patterns

    Particularly effective for medical images with both periodic scan artifacts
    and random local noise.
    """

    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

        # Learnable frequency filter (from FreqEnhancedMona)
        self.freq_filter = nn.Parameter(torch.ones(in_features))

        # Noise estimation and attention (from NoiseAwareMona)
        self.noise_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, in_features // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 4, 3, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        identity = x

        # Step 1: Frequency domain filtering (global periodic artifacts)
        b, c, h, w = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        x_freq = x_freq * self.freq_filter.view(1, -1, 1, 1)
        x = torch.fft.irfft2(x_freq, s=(h, w), dim=(-2, -1))

        # Step 2: Multi-scale convolution
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        # Step 3: Adaptive weighting based on noise level (local noise handling)
        weights = self.noise_estimator(x)  # (B, 3, 1, 1)
        w1 = weights[:, 0:1, :, :]
        w2 = weights[:, 1:2, :, :]
        w3 = weights[:, 2:3, :, :]

        x = conv1_x * w1 + conv2_x * w2 + conv3_x * w3 + identity
        identity = x
        x = self.projector(x)
        return identity + x


class HybridNoiseFreqMona(nn.Module):
    """
    Hybrid MONA adapter combining noise-aware and frequency-enhanced variants

    Internally works with [B, N, D] format (batch-first).
    Accepts [N, B, D] input from OpenAI CLIP and converts it internally.

    This variant provides comprehensive noise handling:
    - Frequency-domain filtering for periodic artifacts
    - Adaptive attention for local noise patterns
    - Multi-scale spatial feature extraction
    """

    def __init__(self, in_dim, bottleneck_dim=64):
        super().__init__()
        self.project1 = nn.Linear(in_dim, bottleneck_dim)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(bottleneck_dim, in_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.adapter_conv = HybridNoiseFreqMonaOp(bottleneck_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        """
        Args:
            x: (N, B, D) - Sequence-first format from OpenAI CLIP
            hw_shapes: (H, W) spatial dimensions
        Returns:
            output: (N, B, D) - Sequence-first format
        """
        x = x.permute(1, 0, 2)  # [N, B, D] -> [B, N, D]
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax

        x = self.project1(x)

        b, n, c = x.shape
        if hw_shapes is not None:
            h, w = hw_shapes
            cls_token = x[:, 0:1, :]
            spatial_tokens = x[:, 1:, :]

            spatial_tokens = spatial_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)
            spatial_tokens = self.adapter_conv(spatial_tokens)
            spatial_tokens = spatial_tokens.permute(0, 2, 3, 1).reshape(b, h * w, c)

            x = torch.cat([cls_token, spatial_tokens], dim=1)
        else:
            h = w = int(np.sqrt(n))
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
            x = self.adapter_conv(x)
            x = x.permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.nonlinear(x)
        x = self.dropout(x)
        x = self.project2(x)

        result = identity + x
        return result.permute(1, 0, 2)  # [B, N, D] -> [N, B, D]


# ============================================================================
# Injection Functions
# ============================================================================


def inject_mona_variant_to_clip(model, variant="hybrid", bottleneck_dim=64, num_layers=None):
    """
    Inject MONA variant adapters into OpenAI CLIP's vision transformer only.

    Args:
        model: OpenAI CLIP model (uses [N, B, D] sequence-first format)
        variant: MONA variant type (baseline, noise_aware, freq_enhanced, hybrid)
        bottleneck_dim: bottleneck dimension
        num_layers: number of layers to inject (None = all layers)

    Returns:
        model: Modified model with MONA adapters
        mona_count: Number of layers with MONA adapters injected

    Note: OpenAI CLIP uses [N, B, D] format. MONA adapters handle the conversion
    internally, so no wrapper is needed.
    """

    variant_classes = {
        "baseline": BaselineMona,
        "noise_aware": NoiseAwareMona,
        "freq_enhanced": FreqEnhancedMona,
        "hybrid": HybridNoiseFreqMona,
    }

    if variant not in variant_classes:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variant_classes.keys())}")

    MonaClass = variant_classes[variant]
    mona_count = 0

    # =========================================================================
    # Inject MONA into vision transformer
    # =========================================================================
    if hasattr(model, "visual") and hasattr(model.visual, "transformer"):
        transformer = model.visual.transformer
        embed_dim = transformer.width

        # Calculate patch grid size
        # Handle both input_resolution (older models) and image_size (newer models)
        if hasattr(model.visual, "input_resolution"):
            input_res = model.visual.input_resolution
        elif hasattr(model.visual, "image_size"):
            input_res = (
                model.visual.image_size[0] if isinstance(model.visual.image_size, tuple) else model.visual.image_size
            )
        else:
            raise AttributeError("Model does not have input_resolution or image_size attribute")

        grid_size = input_res // model.visual.conv1.kernel_size[0]
        hw_shapes = (grid_size, grid_size)

        if hasattr(transformer, "resblocks"):
            blocks = transformer.resblocks
            layers_to_inject = len(blocks) if num_layers is None else min(num_layers, len(blocks))

            for i in range(layers_to_inject):
                # Create MONA adapter - directly use it for OpenAI CLIP [N, B, D] format
                mona = MonaClass(embed_dim, bottleneck_dim)

                # Store in block
                blocks[i].mona = mona

                # Save original forward method
                original_forward = blocks[i].forward

                # Create new forward method with MONA
                def make_forward_with_mona(block, orig_forward):
                    def forward_with_mona(x, **kwargs):
                        x = orig_forward(x, **kwargs)
                        x = block.mona(x, hw_shapes)
                        return x

                    return forward_with_mona

                # Replace forward method
                blocks[i].forward = make_forward_with_mona(blocks[i], original_forward)
                mona_count += 1

    print(f"✓ Injected {variant} MONA adapters to {mona_count} layers (OpenAI CLIP vision encoder)")
    return model, mona_count


def inject_mona_variant_to_open_clip(model, variant="hybrid", bottleneck_dim=64, num_layers=None):
    """
    Inject MONA variant adapters into open_clip-based models' vision encoder only.
    Supported models: MetaCLIP, BiomedCLIP, UniMedCLIP

    Args:
        model: open_clip-based model (uses [B, N, D] batch-first format)
        variant: MONA variant type (baseline, noise_aware, freq_enhanced, hybrid)
        bottleneck_dim: bottleneck dimension
        num_layers: number of layers to inject (None = all layers)

    Returns:
        model: Modified model with MONA adapters
        mona_count: Number of layers with MONA adapters injected

    Note: open_clip models use [B, N, D] format. MONA adapters expect [N, B, D],
    so BatchFirstMonaWrapper is used to handle the conversion.
    """

    variant_classes = {
        "baseline": BaselineMona,
        "noise_aware": NoiseAwareMona,
        "freq_enhanced": FreqEnhancedMona,
        "hybrid": HybridNoiseFreqMona,
    }

    if variant not in variant_classes:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variant_classes.keys())}")

    MonaClass = variant_classes[variant]
    mona_count = 0

    # =========================================================================
    # Inject MONA into vision transformer
    # =========================================================================
    if hasattr(model, "visual"):
        visual = model.visual
        blocks = None
        embed_dim = None
        hw_shapes = None

        # Case 1: BiomedCLIP/UniMedCLIP structure (visual.trunk.blocks)
        if hasattr(visual, "trunk"):
            trunk = visual.trunk
            embed_dim = trunk.embed_dim

            # Calculate patch grid size for spatial convolution
            num_patches = trunk.patch_embed.num_patches
            grid_size = int(np.sqrt(num_patches))
            hw_shapes = (grid_size, grid_size)

            if hasattr(trunk, "blocks"):
                blocks = trunk.blocks

        # Case 2: MetaCLIP structure (visual.transformer.resblocks)
        elif hasattr(visual, "transformer"):
            transformer = visual.transformer
            embed_dim = transformer.width

            # Calculate patch grid size from visual module
            if hasattr(visual, "grid_size"):
                grid_size = visual.grid_size[0]  # Assuming square grid
                hw_shapes = (grid_size, grid_size)
            elif hasattr(visual, "image_size") and hasattr(visual, "patch_size"):
                img_size = visual.image_size[0] if isinstance(visual.image_size, tuple) else visual.image_size
                patch_size = visual.patch_size[0] if isinstance(visual.patch_size, tuple) else visual.patch_size
                grid_size = img_size // patch_size
                hw_shapes = (grid_size, grid_size)

            if hasattr(transformer, "resblocks"):
                blocks = transformer.resblocks

        # Inject MONA adapters if we found blocks
        if blocks is not None and embed_dim is not None:
            layers_to_inject = len(blocks) if num_layers is None else min(num_layers, len(blocks))

            for i in range(layers_to_inject):
                # Create MONA adapter (sequence-first format)
                mona_adapter = MonaClass(embed_dim, bottleneck_dim)
                # Wrap for batch-first format
                mona = BatchFirstMonaWrapper(mona_adapter)

                # Store in block
                blocks[i].mona = mona

                # Save original forward method
                original_forward = blocks[i].forward

                # Create new forward method with MONA
                def make_forward_with_mona(block, orig_forward):
                    def forward_with_mona(x, **kwargs):
                        x = orig_forward(x, **kwargs)
                        x = block.mona(x, hw_shapes)
                        return x

                    return forward_with_mona

                # Replace forward method
                blocks[i].forward = make_forward_with_mona(blocks[i], original_forward)
                mona_count += 1

    print(f"✓ Injected {variant} MONA adapters to {mona_count} layers (open_clip vision encoder)")
    return model, mona_count
