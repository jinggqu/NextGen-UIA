import torch
import torch.nn as nn
import math


class TimmCLIPAdapter(nn.Module):
    """
    Adapter to extract multi-level features from Timm CLIP and convert them to segmentation masks
    """

    def __init__(
        self,
        clip_model,
        extract_layers=[3, 6, 9],
        reduce_dim=512,
        num_classes=2,
        img_size=224,
        patch_size=16,
        task="seg",
    ):
        super(TimmCLIPAdapter, self).__init__()
        self.clip_model = clip_model
        self.extract_layers = extract_layers
        self.reduce_dim = reduce_dim
        self.num_classes = num_classes
        self.task = task

        self.feature_dim = 768  # clip_model.visual.trunk

        # Feature reduction layers - one for each extracted layer
        self.reduces = nn.ModuleList([nn.Linear(self.feature_dim, reduce_dim) for _ in range(len(extract_layers))])

        # Feature processing blocks
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(reduce_dim),
                    nn.Linear(reduce_dim, reduce_dim),
                    nn.GELU(),
                    nn.Linear(reduce_dim, reduce_dim),
                )
                for _ in range(len(extract_layers))
            ]
        )

        self.seg_head = nn.Sequential(
            nn.Upsample((img_size, img_size), mode="bilinear", align_corners=False),
            nn.Conv2d(reduce_dim, num_classes, kernel_size=1),
        )

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, reduce_dim, 1, 1)
            nn.Flatten(),  # (B, reduce_dim)
            nn.Dropout(0.5),
            nn.Linear(reduce_dim, num_classes),
        )

    def extract_vit_features(self, x):
        """Extract features from Vision Transformer layers for Timm CLIP model"""
        visual = self.clip_model.visual

        # Support both model structures:
        # - BiomedCLIP/UniMedCLIP: visual.trunk (timm-style)
        # - MetaCLIP: visual.transformer (OpenAI CLIP-style)
        if hasattr(visual, "trunk"):
            vit = visual.trunk
            blocks = vit.blocks
            has_trunk = True
        elif hasattr(visual, "transformer"):
            # MetaCLIP structure
            vit = visual
            blocks = visual.transformer.resblocks
            has_trunk = False
        else:
            raise AttributeError("Model visual encoder has neither 'trunk' nor 'transformer' attribute")

        # patch embedding
        if has_trunk:
            x = vit.patch_embed(x)
            # add cls token and position embedding
            cls_token = vit.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, embed_dim]
            x = torch.cat((cls_token, x), dim=1)
            x = x + vit.pos_embed
            x = vit.pos_drop(x)
        else:
            # MetaCLIP structure (follows OpenAI CLIP)
            x = vit.conv1(x)  # [B, hidden_dim, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, hidden_dim, grid**2]
            x = x.permute(0, 2, 1)  # [B, grid**2, hidden_dim]

            # add cls token and position embedding
            # class_embedding is [hidden_dim], need to expand to [B, 1, hidden_dim]
            cls_token = vit.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([cls_token, x], dim=1)  # [B, grid**2 + 1, hidden_dim]
            x = x + vit.positional_embedding.to(x.dtype)
            x = vit.ln_pre(x)

        # save activations
        activations = []

        # through transformer blocks
        for i, block in enumerate(blocks):
            x = block(x)
            if i in self.extract_layers:
                activations.append(x.clone())

        # apply final LayerNorm
        if has_trunk:
            x = vit.norm(x)
        else:
            x = vit.ln_post(x[:, 0, :])  # Only apply to CLS token for MetaCLIP

        # return activations
        return x, activations

    def forward(self, x):
        _, activations = self.extract_vit_features(x)

        # Process activations in reverse order (deep to shallow)
        _activations = activations[::-1]
        _reduces = self.reduces[::-1]
        _blocks = self.blocks[::-1]

        # Feature pyramid processing
        a = None
        for i, (activation, reduce, block) in enumerate(zip(_activations, _reduces, _blocks)):
            # for Timm ViT, activation is [batch, tokens, features]

            # Remove CLS token and process
            act_without_cls = activation[:, 1:, :]

            # Reduce dimension then process with block
            reduced_act = reduce(act_without_cls)
            reduced_act = block(reduced_act)

            # Fuse with previous pyramid level
            if a is not None:
                a = reduced_act + a
            else:
                a = reduced_act

        # Reshape to 2D feature map
        batch_size = x.shape[0]

        # For ViT, determine grid size from number of tokens
        size = int(math.sqrt(a.shape[1]))
        a = a.permute(0, 2, 1)  # [batch, features, tokens]
        a = a.view(batch_size, self.reduce_dim, size, size)

        # Apply segmentation or classification head
        if self.task == "seg":
            a = self.seg_head(a)
        elif self.task == "cls":
            a = self.cls_head(a)
        else:
            raise ValueError(f"Invalid task type: {self.task}")

        return a

    def freeze_clip_backbone(self):
        """
        Freeze CLIP model and only train the adapter heads.
        Keeps adapters, MONA, and LoRA modules (if present) trainable.
        """
        # Freeze CLIP backbone, but keep adapters, MONA, and LoRA trainable
        for name, param in self.clip_model.named_parameters():
            if "adapter" in name or "mona" in name or "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Keep adapter modules trainable
        for param in self.reduces.parameters():
            param.requires_grad = True

        for param in self.blocks.parameters():
            param.requires_grad = True

        # Keep task-specific head trainable
        if self.task == "seg":
            for param in self.seg_head.parameters():
                param.requires_grad = True
        elif self.task == "cls":
            for param in self.cls_head.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Invalid task type: {self.task}")
