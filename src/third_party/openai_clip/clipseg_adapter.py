import torch
import torch.nn as nn
import os  # Import os module

# Import necessary components from transformers
from transformers.models.clipseg.modeling_clipseg import CLIPSegForImageSegmentation


class CLIPSegAdapter(nn.Module):
    """
    Adapter using CLIP ViT as encoder and CLIPSegDecoder for segmentation,
    allowing extraction from specific intermediate layers.
    Downloads/loads decoder weights from/to a specified ckpt directory.
    """

    def __init__(
        self,
        clip_model,
        decoder_base_config="CIDAS/clipseg-rd64-refined",
        ckpt_dir="./ckpt",
    ):
        super(CLIPSegAdapter, self).__init__()
        self.clip_model = clip_model
        self.ckpt_dir = ckpt_dir

        # Ensure the checkpoint directory exists
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Load the full pretrained model first to get a fully configured decoder
        pretrained_model = CLIPSegForImageSegmentation.from_pretrained(
            decoder_base_config,
            cache_dir=self.ckpt_dir,
        )

        # Use the fully configured decoder from the pretrained model
        self.decoder = pretrained_model.decoder
        self.extract_layers = self.decoder.config.extract_layers

        # We don't need the full model anymore
        del pretrained_model

    def extract_vit_features(self, x):
        """Extract hidden states from specified Vision Transformer layers."""
        # Initial processing
        x = self.clip_model.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # Add class token and position embedding
        class_embedding = self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_embedding, x], dim=1)
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        # Pass through transformer blocks
        x = x.permute(1, 0, 2)  # NLD -> LND

        extracted_hidden_states = ()  # Use a tuple to store the desired hidden states

        # Iterate through all blocks, storing states only from specified layers
        for i, block in enumerate(self.clip_model.visual.transformer.resblocks):
            x = block(x)
            if i in self.extract_layers:
                # Store hidden state (in NLD format for the decoder)
                current_hidden_state = x.permute(1, 0, 2)  # LND -> NLD
                extracted_hidden_states = extracted_hidden_states + (current_hidden_state,)

        # Return the tuple of selected hidden states
        return extracted_hidden_states

    def forward(self, x, input_ids=None):
        B, _, H, W = x.shape

        # Extract hidden states from the specified ViT encoder layers
        encoder_specific_hidden_states = self.extract_vit_features(x)

        # Get text features from the CLIP model
        conditional_embeddings = self.clip_model.encode_text(input_ids)

        # Pass the selected hidden states AND conditional embeddings to the CLIPSeg Decoder
        decoder_output = self.decoder(
            hidden_states=encoder_specific_hidden_states,  # Pass the tuple of selected states
            conditional_embeddings=conditional_embeddings,
        )
        logits = decoder_output[0]

        # Reshape to match the original spatial dimensions and ensure 2 channels for binary segmentation
        logits = logits.view(B, -1, H, W)

        # If the decoder outputs single channel, expand to 2 channels for binary classification
        if logits.shape[1] == 1:
            # Create background and foreground logits
            background_logits = -logits  # Inverse of foreground
            logits = torch.cat([background_logits, logits], dim=1)  # [B, 2, H, W]

        return logits

    def freeze_clip_backbone(self):
        """
        Freeze CLIP model and only train the decoder head.
        """
        # Freeze CLIP encoder parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Unfreeze Decoder parameters
        for param in self.decoder.parameters():
            param.requires_grad = True
