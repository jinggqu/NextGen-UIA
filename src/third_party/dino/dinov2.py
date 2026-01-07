# This file is modified from https://github.com/MohammedSB/DINOv2ForRadiology/blob/main/dinov2/eval/segmentation/utils.py#L12

from urllib.parse import urlparse
import torch
import torch.nn as nn

from torchvision.transforms import transforms


class DINOV2Encoder(torch.nn.Module):
    def __init__(self, encoder, n_last_blocks=1) -> None:
        super(DINOV2Encoder, self).__init__()
        self.encoder = encoder
        self.encoder.eval()
        self.n_last_blocks = n_last_blocks

    def forward(self, x):
        """
        Returns features following official DINOv2 format:
        - If n_last_blocks=1: dict with 'x_norm_clstoken' and 'x_norm_patchtokens'
        - If n_last_blocks>1: list of (patch_tokens, cls_token) tuples
        """
        if self.n_last_blocks == 1:
            # Single layer: return full dict with both cls token and patch tokens
            return self.encoder.forward_features(x)
        else:
            # Multiple layers: return list of (patch_tokens, cls_token) tuples
            # Official format: [(patch_tokens, cls_token), ...] for each layer
            features = self.encoder.get_intermediate_layers(x, self.n_last_blocks, return_class_token=True)
            return features


class ClassificationHead(nn.Module):
    """
    Linear classification head following DINOv2 official implementation.
    Official approach: Concatenates CLS token(s) and averaged patch tokens before linear projection.
    Reference: https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/classifiers.py#L58-L83
    """

    def __init__(self, embed_dim, num_classes=2, layers=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.layers = layers

        # Official DINOv2 uses: cls_token + avg_patch_tokens
        # For 1 layer: [cls_token, avg_patch] = 2 * embed_dim
        # For 4 layers: [cls1, cls2, cls3, cls4, avg_patch4] = 5 * embed_dim
        if layers == 1:
            linear_input_dim = 2 * embed_dim
        elif layers == 4:
            linear_input_dim = 5 * embed_dim
        else:
            raise ValueError(f"Unsupported number of layers: {layers}. Only 1 and 4 are supported.")

        self.linear = nn.Linear(linear_input_dim, num_classes)
        # Initialize weights following official implementation
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, features):
        """
        Args:
            features:
                - If layers=1: dict with keys 'x_norm_clstoken' and 'x_norm_patchtokens'
                    * x_norm_clstoken: [B, embed_dim]
                    * x_norm_patchtokens: [B, num_patches, embed_dim]
                - If layers=4: list of 4 tuples [(patch_tokens_i, cls_token_i), ...]
                    * Each patch_tokens_i: [B, num_patches, embed_dim]
                    * Each cls_token_i: [B, embed_dim]
        Returns:
            logits with shape [B, num_classes]
        """
        if self.layers == 1:
            # Single layer: use dict format from forward_features()
            cls_token = features["x_norm_clstoken"]  # [B, embed_dim]
            patch_tokens = features["x_norm_patchtokens"]  # [B, num_patches, embed_dim]

            # Global average pooling over patch tokens (official approach)
            avg_patch_tokens = patch_tokens.mean(dim=1)  # [B, embed_dim]

            # Concatenate CLS token and averaged patch tokens
            linear_input = torch.cat([cls_token, avg_patch_tokens], dim=1)  # [B, 2*embed_dim]

        elif self.layers == 4:
            # Multiple layers: use list format from get_intermediate_layers()
            # features is a list of 4 tuples: [(patch_tokens, cls_token), ...]
            # Official: concatenate [cls1, cls2, cls3, cls4, avg_patch4]
            linear_input = torch.cat(
                [
                    features[0][1],  # cls_token from layer 1
                    features[1][1],  # cls_token from layer 2
                    features[2][1],  # cls_token from layer 3
                    features[3][1],  # cls_token from layer 4
                    features[3][0].mean(dim=1),  # avg patch_tokens from layer 4
                ],
                dim=1,
            )  # [B, 5*embed_dim]

        return self.linear(linear_input)  # [B, num_classes]


class LinearDecoder(torch.nn.Module):
    """Linear decoder head"""

    DECODER_TYPE = "linear"

    def __init__(self, in_channels, num_classes=3, image_size=448, patch_size=14):
        super().__init__()
        print(patch_size)
        self.image_size = image_size
        self.in_channels = in_channels
        self.width = self.height = image_size // patch_size
        self.decoder = torch.nn.Conv2d(in_channels, num_classes, (1, 1))
        self.decoder.weight.data.normal_(mean=0.0, std=0.01)
        self.decoder.bias.data.zero_()

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        output = self.decoder(embeddings)

        # Upsample (interpolate) output/logit map.
        output = torch.nn.functional.interpolate(output, size=self.image_size, mode="bilinear", align_corners=False)

        return output


class UNetDecoderUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=1024) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.skip_conv = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x2 = self.skip_conv(x2)
        scale_factor = x1.size()[2] / x2.size()[2]
        x2 = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)(x2)
        x = torch.concat([x1, x2], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """Unet decoder head"""

    DECODER_TYPE = "unet"

    def __init__(self, in_channels, out_channels, image_size=224, resize_image=False, patch_size=14):
        super(UNetDecoder, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = in_channels
        self.image_size = image_size
        self.resize_image = resize_image
        self.up1 = UNetDecoderUpBlock(in_channels=in_channels, out_channels=in_channels // 2, embed_dim=self.embed_dim)
        self.up2 = UNetDecoderUpBlock(
            in_channels=in_channels // 2, out_channels=in_channels // 4, embed_dim=self.embed_dim
        )
        self.up3 = UNetDecoderUpBlock(
            in_channels=in_channels // 4, out_channels=in_channels // 8, embed_dim=self.embed_dim
        )
        self.up4 = UNetDecoderUpBlock(in_channels=in_channels // 8, out_channels=out_channels, embed_dim=self.embed_dim)

    def forward(self, x):
        """
        Args:
            x: List of features from get_intermediate_layers()
               Format: [(patch_tokens, cls_token), ...] for n_last_blocks > 1
               Each patch_tokens shape: [B, num_patches, embed_dim]
        """
        h = w = self.image_size // self.patch_size

        # Extract patch tokens from tuples (ignore cls tokens for segmentation)
        # x is a list of (patch_tokens, cls_token) tuples
        skip1 = x[3][0].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)
        skip2 = x[2][0].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)
        skip3 = x[1][0].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)
        skip4 = x[0][0].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)
        x1 = x[4][0].reshape(-1, h, w, self.embed_dim).permute(0, 3, 1, 2)

        x2 = self.up1(x1, skip1)
        x3 = self.up2(x2, skip2)
        x4 = self.up3(x3, skip3)
        x5 = self.up4(x4, skip4)

        if self.resize_image:
            x5 = transforms.Resize(
                (self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC
            )(x5)
        return x5


class LinearPostprocessor(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, samples, targets):
        logits = self.decoder(samples)
        if isinstance(logits, list) or (isinstance(logits, torch.Tensor) and len(logits.size()) > 4):  # if 3D output
            logits = torch.cat(logits, dim=0)
            targets = torch.cat(targets, dim=0).cuda()

        preds = logits.argmax(dim=1)
        targets = targets.type(torch.int64)

        return {
            "preds": preds,
            "target": targets,
        }


class AllDecoders(nn.Module):
    def __init__(self, decoders_dict, decoder_type):
        super().__init__()
        self.decoders_dict = nn.ModuleDict()
        self.decoders_dict.update(decoders_dict)
        self.decoder_type = decoder_type

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.decoders_dict.items()}

    def __len__(self):
        return len(self.decoders_dict)


def setup_decoders(embed_dim, learning_rates, num_classes=14, decoder_type="linear", image_size=224, patch_size=14):
    """
    Sets up the multiple segmentors with different hyperparameters to test out the most optimal one
    """
    decoders_dict = nn.ModuleDict()
    optim_param_groups = []
    for lr in learning_rates:
        if decoder_type == "linear":
            decoder = LinearDecoder(embed_dim, num_classes=num_classes, image_size=image_size, patch_size=patch_size)
        elif decoder_type == "unet":
            decoder = UNetDecoder(
                in_channels=embed_dim,
                out_channels=num_classes,
                image_size=image_size,
                resize_image=True,
                patch_size=patch_size,
            )
        decoder = decoder.cuda()
        decoders_dict[f"{decoder_type}:lr={lr:.10f}".replace(".", "_")] = decoder
        optim_param_groups.append({"params": decoder.parameters(), "lr": lr})

    decoders = AllDecoders(decoders_dict, decoder_type)

    return decoders, optim_param_groups


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        print(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]

    state_dict = {"encoder." + k: v for k, v in state_dict.items()}
    state_dict = {k.replace("blocks.", "blocks.0."): v for k, v in state_dict.items()}

    msg = model.load_state_dict(state_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))
