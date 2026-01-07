"""
LoRA Adapter for CLIP Fine-tuning

Implements Low-Rank Adaptation (LoRA) for CLIP models.
Based on the simple and reliable LoRA implementation from clip_lora.py
"""

import math
import torch
import torch.nn as nn


class LoRALayer:
    """Base LoRA Layer with common functionality"""

    def __init__(self, r: int, lora_alpha: int, dropout_rate: float = 0):
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout_rate
        if self.r > 0:
            self.scaling = self.lora_alpha / math.sqrt(self.r)
        self.merged = False
        self.params_with_lora = {}

    def register_lora_param(self):
        """Register LoRA matrices A and B"""
        for param_name, lora_name in self.params_with_lora.items():
            assert len(eval(f"self.{param_name}").size()) == 2
            self.register_parameter(
                f"{lora_name}_lora_A",
                nn.Parameter(eval(f"self.{param_name}").new_zeros((self.r, eval(f"self.{param_name}").size()[1]))),
            )
            self.register_parameter(
                f"{lora_name}_lora_B",
                nn.Parameter(eval(f"self.{param_name}").new_zeros((eval(f"self.{param_name}").size()[0], self.r))),
            )
            eval(f"self.{param_name}").requires_grad = False

    def init_lora_param(self):
        """Initialize LoRA parameters"""
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f"{lora_name}_lora_A"):
                nn.init.kaiming_uniform_(eval(f"self.{lora_name}_lora_A"), a=math.sqrt(5))
                nn.init.zeros_(eval(f"self.{lora_name}_lora_B"))

    def merge_BA(self, param_name: str):
        """Merge B @ A matrices"""
        lora_name = self.params_with_lora[param_name]
        return (eval(f"self.{lora_name}_lora_B") @ eval(f"self.{lora_name}_lora_A")).view(
            eval(f"self.{param_name}").shape
        )


class LinearLoRA(nn.Linear, LoRALayer):
    """LoRA implemented in a Linear layer"""

    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__(in_features=existing_linear.in_features, out_features=existing_linear.out_features)
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)

        self.params_with_lora = {"weight": "w"}
        if r > 0:
            self.register_lora_param()
        self.init_lora_param()

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor):
        # Compute the original linear transformation
        original_output = nn.Linear.forward(self, x)

        if self.training and self.dropout is not None and self.dropout.p > 0:
            x = self.dropout(x)

        if self.r > 0:
            lora_adjustment = torch.matmul(x, self.merge_BA("weight").T) * self.scaling
            result = original_output + lora_adjustment
        else:
            result = original_output
        return result


class PlainMultiheadAttentionLoRA(nn.Module):
    """MultiheadAttention with LoRA adapters"""

    def __init__(
        self,
        existing_mha: nn.MultiheadAttention,
        enable_lora: list = ["q", "k", "v", "o"],
        r: int = 0,
        lora_alpha: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.dropout = 0
        self.embed_dim = existing_mha.embed_dim
        self.kdim = existing_mha.kdim
        self.vdim = existing_mha.vdim
        self._qkv_same_embed_dim = existing_mha._qkv_same_embed_dim
        self.num_heads = existing_mha.num_heads
        self.batch_first = existing_mha.batch_first
        self.head_dim = existing_mha.head_dim

        # Create separate Q, K, V projection layers
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None)

        # Initialize parameters from existing MHA
        with torch.no_grad():
            existing_weight = existing_mha.in_proj_weight.data
            existing_bias = existing_mha.in_proj_bias.data if existing_mha.in_proj_bias is not None else None

            # Split and copy Q, K, V weights
            self.q_proj.weight.data.copy_(existing_weight[: self.embed_dim, :])
            if existing_bias is not None:
                self.q_proj.bias.data.copy_(existing_bias[: self.embed_dim])

            self.k_proj.weight.data.copy_(existing_weight[self.embed_dim : 2 * self.embed_dim, :])
            if existing_bias is not None:
                self.k_proj.bias.data.copy_(existing_bias[self.embed_dim : 2 * self.embed_dim])

            self.v_proj.weight.data.copy_(existing_weight[2 * self.embed_dim :, :])
            if existing_bias is not None:
                self.v_proj.bias.data.copy_(existing_bias[2 * self.embed_dim :])

            # Copy output projection weights
            self.proj.weight.data.copy_(existing_mha.out_proj.weight.data)
            if self.proj.bias is not None:
                self.proj.bias.data.copy_(existing_mha.out_proj.bias.data)

        # Apply LoRA to selected components
        for item in enable_lora:
            if item == "q":
                self.q_proj = LinearLoRA(self.q_proj, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)
            elif item == "k":
                self.k_proj = LinearLoRA(self.k_proj, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)
            elif item == "v":
                self.v_proj = LinearLoRA(self.v_proj, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)
            elif item == "o":
                self.proj = LinearLoRA(self.proj, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, **kwargs):
        """Forward pass with LoRA-adapted attention"""
        is_batched = query.dim() == 3

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        # Apply Q, K, V projections with LoRA
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        # Scaled dot-product attention
        dropout_p = self.dropout if self.training else 0.0
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
        )

        # Reshape and apply output projection with LoRA
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None


def inject_lora_to_clip(model, lora_r=16, lora_alpha=32, lora_dropout=0.1, num_layers=None):
    """
    Inject LoRA adapters into CLIP's vision transformer only.

    Note: LoRA is only injected into the vision encoder, not the text encoder.
    This is because our downstream tasks (classification, segmentation) primarily
    rely on visual features, and keeping the text encoder frozen ensures fair
    comparison with MONA which is designed specifically for visual features.

    Args:
        model: CLIP model
        lora_r: LoRA rank (bottleneck dimension)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        num_layers: Number of layers to inject LoRA (None = all layers)

    Returns:
        model: Modified model with LoRA adapters
        lora_count: Number of layers with LoRA adapters injected
    """
    lora_count = 0
    enable_lora = ["q", "k", "v", "o"]  # Apply LoRA to Q, K, V, and output projections

    # Inject LoRA into vision transformer only
    if hasattr(model, "visual") and hasattr(model.visual, "transformer"):
        transformer = model.visual.transformer
        if hasattr(transformer, "resblocks"):
            blocks = transformer.resblocks
            layers_to_inject = len(blocks) if num_layers is None else min(num_layers, len(blocks))

            for i in range(layers_to_inject):
                block = blocks[i]

                # Replace MultiheadAttention with LoRA version
                if hasattr(block, "attn") and isinstance(block.attn, nn.MultiheadAttention):
                    new_attn = PlainMultiheadAttentionLoRA(
                        block.attn,
                        enable_lora=enable_lora,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        dropout_rate=lora_dropout,
                    )
                    block.attn = new_attn
                    lora_count += 1

    print(f"✓ Injected LoRA adapters to {lora_count} layers (CLIP vision encoder)")
    return model, lora_count


def inject_lora_to_biomedclip(
    model,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    num_layers=None,
    tune_text_encoder=False,
):
    """
    Inject LoRA adapters into BiomedCLIP's vision transformer only.

    Note: LoRA is only injected into the vision encoder, not the text encoder.
    This is because our downstream tasks (classification, segmentation) primarily
    rely on visual features, and keeping the text encoder frozen ensures fair
    comparison with MONA which is designed specifically for visual features.

    BiomedCLIP uses:
    - Vision: timm VisionTransformer with Attention modules (qkv + proj)

    Args:
        model: BiomedCLIP model
        lora_r: LoRA rank (bottleneck dimension)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        num_layers: Number of layers to inject LoRA (None = all layers)
        inject_lora_to_text_encoder: Whether to inject LoRA adapters into the text encoder
    Returns:
        model: Modified model with LoRA adapters
        lora_count: Number of layers with LoRA adapters injected
    """
    lora_count = 0

    # Inject LoRA into vision transformer only (timm-style VisionTransformer)
    if hasattr(model, "visual") and hasattr(model.visual, "trunk"):
        trunk = model.visual.trunk
        if hasattr(trunk, "blocks"):
            blocks = trunk.blocks
            layers_to_inject = len(blocks) if num_layers is None else min(num_layers, len(blocks))

            for i in range(layers_to_inject):
                block = blocks[i]

                # Replace attention QKV and projection with LoRA versions
                if hasattr(block, "attn"):
                    attn = block.attn

                    # Apply LoRA to QKV projection
                    if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                        attn.qkv = LinearLoRA(
                            attn.qkv,
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            dropout_rate=lora_dropout,
                        )

                    # Apply LoRA to output projection
                    if hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
                        attn.proj = LinearLoRA(
                            attn.proj,
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            dropout_rate=lora_dropout,
                        )

                    lora_count += 1

    if tune_text_encoder:
        # Inject LoRA into text transformer (BERT-style)
        if hasattr(model, "text") and hasattr(model.text, "transformer"):
            transformer = model.text.transformer
            if hasattr(transformer, "encoder") and hasattr(transformer.encoder, "layer"):
                layers = transformer.encoder.layer
                layers_to_inject = len(layers) if num_layers is None else min(num_layers, len(layers))

                for i in range(layers_to_inject):
                    layer = layers[i]

                    # Replace BERT attention with LoRA versions
                    if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
                        self_attn = layer.attention.self

                        # Apply LoRA to query, key, value projections
                        if hasattr(self_attn, "query") and isinstance(self_attn.query, nn.Linear):
                            self_attn.query = LinearLoRA(
                                self_attn.query,
                                r=lora_r,
                                lora_alpha=lora_alpha,
                                dropout_rate=lora_dropout,
                            )

                        if hasattr(self_attn, "key") and isinstance(self_attn.key, nn.Linear):
                            self_attn.key = LinearLoRA(
                                self_attn.key,
                                r=lora_r,
                                lora_alpha=lora_alpha,
                                dropout_rate=lora_dropout,
                            )

                        if hasattr(self_attn, "value") and isinstance(self_attn.value, nn.Linear):
                            self_attn.value = LinearLoRA(
                                self_attn.value,
                                r=lora_r,
                                lora_alpha=lora_alpha,
                                dropout_rate=lora_dropout,
                            )

                        # Apply LoRA to output projection
                        if hasattr(layer.attention, "output") and hasattr(layer.attention.output, "dense"):
                            if isinstance(layer.attention.output.dense, nn.Linear):
                                layer.attention.output.dense = LinearLoRA(
                                    layer.attention.output.dense,
                                    r=lora_r,
                                    lora_alpha=lora_alpha,
                                    dropout_rate=lora_dropout,
                                )

                        lora_count += 1

    print(f"✓ Injected LoRA adapters to {lora_count} layers (BiomedCLIP vision encoder)")
    return model, lora_count
