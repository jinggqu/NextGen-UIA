#!/usr/bin/env bash

mkdir -p logs

{
    # ========== Fine-tuning with Refined Caption ==========
    python clip_finetune.py --caption_type refined
    mv runs/clip_mona_finetune/refined.pth ckpt/clip_mona_finetune_refined.pth

    # ========== Zero-shot Classification ==========
    # CLIP (Mona, fine-tuned)
    python clip_cls_zero_shot.py --caption_type refined --finetuned_weights ckpt/clip_mona_finetune_refined.pth --dataset LN-1
    python clip_cls_zero_shot.py --caption_type refined --finetuned_weights ckpt/clip_mona_finetune_refined.pth --dataset LN-2
    python clip_cls_zero_shot.py --caption_type refined --finetuned_weights ckpt/clip_mona_finetune_refined.pth --dataset BUSI

    # ========== Supervised Segmentation ==========
    # CLIP (Mona, fine-tuned)
    python clip_seg.py --caption_type refined --finetuned_weights ckpt/clip_mona_finetune_refined.pth --dataset LN-1
    mkdir -p runs/clip_mona_seg/LN-2/train/clip_mona_seg_mona_refined
    cp runs/clip_mona_seg/LN-1/train/clip_mona_seg_mona_refined/adapter_components.pth runs/clip_mona_seg/LN-2/train/clip_mona_seg_mona_refined/adapter_components.pth
    python clip_seg.py --caption_type refined --finetuned_weights ckpt/clip_mona_finetune_refined.pth --dataset LN-2 --test
    python clip_seg.py --caption_type refined --finetuned_weights ckpt/clip_mona_finetune_refined.pth --dataset BUSI
    python clip_seg.py --caption_type refined --finetuned_weights ckpt/clip_mona_finetune_refined.pth --dataset DDTI
    python clip_seg.py --caption_type refined --finetuned_weights ckpt/clip_mona_finetune_refined.pth --dataset TN3K
    python clip_seg.py --caption_type refined --finetuned_weights ckpt/clip_mona_finetune_refined.pth --dataset microUS

    # CLIP (Mona)
    python clip_seg.py --dataset LN-1
    mkdir -p runs/clip_mona_seg/LN-2/train/clip_mona_seg_raw_na
    cp runs/clip_mona_seg/LN-1/train/clip_mona_seg_raw_na/adapter_components.pth runs/clip_mona_seg/LN-2/train/clip_mona_seg_raw_na/adapter_components.pth
    python clip_seg.py --dataset LN-2 --test
    python clip_seg.py --dataset BUSI
    python clip_seg.py --dataset DDTI
    python clip_seg.py --dataset TN3K
    python clip_seg.py --dataset microUS

    # ========== Supervised Classification ==========
    # CLIP (Mona, fine-tuned)
    python clip_cls.py --caption_type refined --finetuned_weights ckpt/clip_mona_finetune_refined.pth --dataset LN-1
    mkdir -p runs/clip_mona_cls/LN-2/train/clip_mona_cls_mona_refined
    cp runs/clip_mona_cls/LN-1/train/clip_mona_cls_mona_refined/adapter_components.pth runs/clip_mona_cls/LN-2/train/clip_mona_cls_mona_refined/adapter_components.pth
    python clip_cls.py --caption_type refined --finetuned_weights ckpt/clip_mona_finetune_refined.pth --dataset LN-2 --test
    python clip_cls.py --caption_type refined --finetuned_weights ckpt/clip_mona_finetune_refined.pth --dataset BUSI

    # CLIP (Mona)
    python clip_cls.py --dataset LN-1
    mkdir -p runs/clip_mona_cls/LN-2/train/clip_mona_cls_raw_na
    cp runs/clip_mona_cls/LN-1/train/clip_mona_cls_raw_na/adapter_components.pth runs/clip_mona_cls/LN-2/train/clip_mona_cls_raw_na/adapter_components.pth
    python clip_cls.py --dataset LN-2 --test
    python clip_cls.py --dataset BUSI

} | tee -a logs/clip_mona.log
