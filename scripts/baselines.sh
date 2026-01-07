#!/usr/bin/env bash
#
# Baseline Models Experiment Pipeline
# Models: ResNet (Classification), UNet (Segmentation)
#

set -e

echo "========================================================================"
echo "Baseline Models Experiment Pipeline"
echo "========================================================================"

mkdir -p logs runs

# Training parameters
SUPERVISED_EPOCHS=200
SEEDS=(1 2 3)

# Datasets
SEG_DATASETS=("LN-INT" "BUSI" "DDTI" "TN3K" "Prostate")
CLS_DATASETS=("LN-INT" "BUSI")

{
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "========================================================================"
        echo "Run $seed / ${#SEEDS[@]}"
        echo "========================================================================"

        # =========================================================================
        # Step 1: ResNet Classification
        # =========================================================================
        echo ""
        echo "[Step 1] ResNet Classification Baseline"
        echo ""

        for dataset in "${CLS_DATASETS[@]}"; do
            python -m src.models.baselines.classification \
                --dataset $dataset \
                --epochs $SUPERVISED_EPOCHS \
                --seed $seed \
                --exp resnet/cls_$seed
        done

        # Using model trained on LN-INT to evaluate on LN-EXT
        mkdir -p runs/resnet/cls_$seed/LN-EXT/train
        cp runs/resnet/cls_$seed/LN-INT/train/best_model.pth runs/resnet/cls_$seed/LN-EXT/train/best_model.pth
        python -m src.models.baselines.classification \
            --dataset LN-EXT \
            --epochs $SUPERVISED_EPOCHS \
            --seed $seed \
            --exp resnet/cls_$seed \
            --test

        # =========================================================================
        # Step 2: UNet Segmentation
        # =========================================================================
        echo ""
        echo "[Step 2] UNet Segmentation Baseline"
        echo ""

        for dataset in "${SEG_DATASETS[@]}"; do
            python -m src.models.baselines.segmentation \
                --dataset $dataset \
                --epochs $SUPERVISED_EPOCHS \
                --seed $seed \
                --exp unet/seg_$seed
        done

        # Using model trained on LN-INT to evaluate on LN-EXT
        mkdir -p runs/unet/seg_$seed/LN-EXT/train
        cp runs/unet/seg_$seed/LN-INT/train/best_model.pth runs/unet/seg_$seed/LN-EXT/train/best_model.pth
        python -m src.models.baselines.segmentation \
            --dataset LN-EXT \
            --epochs $SUPERVISED_EPOCHS \
            --seed $seed \
            --exp unet/seg_$seed \
            --test

        echo ""
        echo "  ✓ Run $seed completed"
    done

    echo ""
    echo "========================================================================"
    echo "Baseline Models Experiment Summary"
    echo "========================================================================"
    echo ""
    echo "Completed:"
    echo "  1. ResNet classification (${#SEEDS[@]} seeds x ${#CLS_DATASETS[@]} datasets)"
    echo "  2. UNet segmentation (${#SEEDS[@]} seeds x ${#SEG_DATASETS[@]} datasets)"
    echo ""

} 2>&1 | tee -a logs/baselines.log
