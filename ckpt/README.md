# Prepare Checkpoints

| **Model (Filename)**                 | **Repo**                                                                                                                                                                     | **Download Link**                                                                                                                                |
| :----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| CLIP visual backbone (ViT-B-16.pt)   | [openai/CLIP](https://github.com/openai/CLIP)                                                                                                                                | [download](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)          |
| DINOv2 (dinov2_vitb14_pretrain.pth)  | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models:~:text=backbone%20only-,ViT%2DB/14%20distilled,86%20M,-%E2%9D%8C)  | [download](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth)                                                       |
| MetaCLIP (metaclip_vit_b16.pt)       | [facebookresearch/MetaCLIP](https://github.com/facebookresearch/MetaCLIP#:~:text=65.5-,ViT%2DB%2D16%2Dquickgelu,-metaclip_400m)                                              | [download](https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_400m.pt)                                                                             |
| UniMed-CLIP (unimed_clip_vit_b16.pt) | [mbzuai-oryx/UniMed-CLIP](https://github.com/mbzuai-oryx/UniMed-CLIP?tab=readme-ov-file#:~:text=on%2021%20datasets-,ViT%2DB%2D16%2Dquickgelu,-BiomedNLP%2DBiomedBERT%2Dbase) | [download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/Ee8EpjZS6SJGiZUrV7DyLxkBrVFir5YzMjYZIc8aEc2oUA?e=I7KvRb) |

The pre-trained models should be placed in the current `ckpt` directory, the folder structure should look like this:

```bash
ckpt
├── README.md
├── ViT-B-16.pt
├── dinov2_vitb14_pretrain.pth
├── metaclip_vit_b16.pt
└── unimed_clip_vit_b16.pt
```