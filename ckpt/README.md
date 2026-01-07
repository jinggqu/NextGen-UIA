# Prepare Checkpoints

## Models Requiring Manual Download

The following models need to be manually downloaded and placed in this `ckpt/` directory:

| **Model (Filename)**                 | **Repo**                                                                                                   | **Download Link**                                                                                                                                |
| :----------------------------------- | ---------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| CLIP visual backbone (ViT-B-16.pt)   | [openai/CLIP](https://github.com/openai/CLIP)                                                              | [download](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)          |
| DINOv2 (dinov2_vitb14_pretrain.pth)  | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models) | [download](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth)                                                       |
| UniMed-CLIP (unimed_clip_vit_b16.pt) | [mbzuai-oryx/UniMed-CLIP](https://github.com/mbzuai-oryx/UniMed-CLIP?tab=readme-ov-file)                   | [download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/Ee8EpjZS6SJGiZUrV7DyLxkBrVFir5YzMjYZIc8aEc2oUA?e=I7KvRb) |

## Models Automatically Downloaded from HuggingFace

The following models will be **automatically downloaded** from HuggingFace when first used:

| **Model**  | **HuggingFace Repo**                                                                                                                        |
| :--------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| BiomedCLIP | [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) |
| MetaCLIP   | [timm/vit_base_patch16_clip_224.metaclip_400m](https://huggingface.co/timm/vit_base_patch16_clip_224.metaclip_400m)                         |
| CLIPSeg    | [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined)                                                             |
| ResNet18   | Automatically downloaded via torchvision                                                                                                    |

## Directory Structure

After downloading the required models, your `ckpt/` directory should look like this:

```bash
ckpt/
├── README.md
├── ViT-B-16.pt                        # OpenAI CLIP (manual download)
├── dinov2_vitb14_pretrain.pth         # DINOv2 (manual download)
├── unimed_clip_vit_b16.pt             # UniMed-CLIP (manual download)
├── checkpoints/                       # Auto-created cache
│   └── resnet18-f37072fd.pth          # ResNet18 (auto-downloaded)
├── models--microsoft--BiomedCLIP...   # BiomedCLIP (auto-downloaded)
├── models--timm--vit_base...          # MetaCLIP (auto-downloaded)
└── models--CIDAS--clipseg...          # CLIPSeg (auto-downloaded)
```

**Note:** The `models--*` directories are HuggingFace cache directories created automatically. You don't need to create them manually.
