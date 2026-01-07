import torch
import torch.nn as nn
import math


class CLIPAdapter(nn.Module):
    """
    Adapter to extract multi-level features from CLIP and convert them to segmentation masks
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
        super(CLIPAdapter, self).__init__()
        self.clip_model = clip_model
        self.extract_layers = extract_layers
        self.reduce_dim = reduce_dim
        self.num_classes = num_classes
        self.task = task

        self.feature_dim = clip_model.visual.transformer.width

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
            nn.Linear(reduce_dim, reduce_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(reduce_dim, num_classes),
        )

    def extract_vit_features(self, x):
        """Extract features from Vision Transformer layers"""
        # Initial processing - similar to the start of VisionTransformer.forward
        x = self.clip_model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # Add class token and position embedding
        x = torch.cat(
            [
                self.clip_model.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )

        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        # Pass through transformer blocks
        x = x.permute(1, 0, 2)  # NLD -> LND
        activations = []

        for i, block in enumerate(self.clip_model.visual.transformer.resblocks):
            x = block(x)
            if i in self.extract_layers:
                activations.append(x.clone())

        # Return the activations
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
            # For ViT, activation is [tokens, batch, features]
            # We need to reshape it to [batch, tokens, features]
            activation = activation.permute(1, 0, 2)

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
        Keeps MONA adapters (if present) trainable.
        """
        # Freeze CLIP backbone, but keep MONA adapters trainable
        for name, param in self.clip_model.named_parameters():
            if "mona" in name:
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
