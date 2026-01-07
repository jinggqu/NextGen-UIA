#!/usr/bin/env bash
#
# BiomedCLIP Complete Experiment Pipeline
#
# This script runs formal experiments comparing:
#   1. Original BiomedCLIP (baseline)
#   2. Hybrid MONA fine-tuned BiomedCLIP
#
# Tasks:
#   - Image-Text Retrieval (ROCO-V2)
#   - Zero-shot Classification (LN-INT, LN-EXT, BUSI)
#   - Supervised Classification (LN-INT, BUSI)
#   - Supervised Segmentation (LN-INT, BUSI, DDTI, TN3K, Prostate)
#

set -e

echo "========================================================================"
echo "BiomedCLIP Complete Experiment Pipeline"
echo "========================================================================"

mkdir -p logs runs

# Training parameters
BATCH_SIZE=64
LR=1e-4
FINETUNE_EPOCHS=32
SUPERVISED_EPOCHS=200
SEEDS=(1 2 3)

# MONA variant - using hybrid (noise_aware + freq_enhanced)
MONA_VARIANT="hybrid"

# Datasets
SEG_DATASETS=("LN-INT" "BUSI" "DDTI" "TN3K" "Prostate")
CLS_DATASETS=("LN-INT" "BUSI")
ZS_DATASETS=("LN-INT" "LN-EXT" "BUSI")

{
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "========================================================================"
        echo "Run $seed / ${#SEEDS[@]}"
        echo "========================================================================"

        # =========================================================================
        # Step 1: Fine-tuning with MONA adapter (Hybrid variant)
        # =========================================================================
        echo ""
        echo "[Step 1] Fine-tuning BiomedCLIP with MONA adapter (${MONA_VARIANT})"
        echo "  - Epochs: $FINETUNE_EPOCHS"
        echo "  - Batch Size: $BATCH_SIZE"
        echo "  - Learning Rate: $LR"
        echo "  - MONA Variant: $MONA_VARIANT"
        echo "  - Seed: $seed"
        echo ""

        python -m src.models.biomedclip.finetune \
            --lr $LR \
            --batch_size $BATCH_SIZE \
            --epochs $FINETUNE_EPOCHS \
            --seed $seed \
            --mona_variant $MONA_VARIANT \
            --exp biomedclip/finetune_mona_${MONA_VARIANT}_$seed

        MONA_FINETUNED_PATH="runs/biomedclip/finetune_mona_${MONA_VARIANT}_$seed/best_model.pth"

        # =========================================================================
        # Step 2: Zero-shot Classification
        # =========================================================================
        echo ""
        echo "[Step 2] Zero-shot Classification"
        echo ""

        # Baseline (without MONA fine-tuning)
        for dataset in "${ZS_DATASETS[@]}"; do
            echo "  [2.1] Zero-shot on $dataset - Baseline"
            python -m src.models.biomedclip.zero_shot \
                --dataset $dataset \
                --seed $seed \
                --exp biomedclip/zero_shot_baseline_$seed
        done

        # With MONA fine-tuned weights
        for dataset in "${ZS_DATASETS[@]}"; do
            echo "  [2.2] Zero-shot on $dataset - MONA ${MONA_VARIANT}"
            python -m src.models.biomedclip.zero_shot \
                --dataset $dataset \
                --mona_weights $MONA_FINETUNED_PATH \
                --mona_variant $MONA_VARIANT \
                --seed $seed \
                --exp biomedclip/zero_shot_mona_${MONA_VARIANT}_$seed
        done

        # =========================================================================
        # Step 3: Supervised Classification
        # =========================================================================
        echo ""
        echo "[Step 3] Supervised Classification"
        echo ""

        # Without MONA (baseline)
        for dataset in "${CLS_DATASETS[@]}"; do
            echo "  [3.1] Classification on $dataset - Baseline"
            python -m src.models.biomedclip.classification \
                --dataset $dataset \
                --epochs $SUPERVISED_EPOCHS \
                --seed $seed \
                --exp biomedclip/cls_baseline_$seed
        done

        # With MONA fine-tuned weights
        for dataset in "${CLS_DATASETS[@]}"; do
            echo "  [3.2] Classification on $dataset - MONA ${MONA_VARIANT}"
            python -m src.models.biomedclip.classification \
                --dataset $dataset \
                --mona_weights $MONA_FINETUNED_PATH \
                --mona_variant $MONA_VARIANT \
                --epochs $SUPERVISED_EPOCHS \
                --seed $seed \
                --exp biomedclip/cls_mona_${MONA_VARIANT}_$seed
        done

        # Cross-dataset evaluation: LN-INT -> LN-EXT
        echo ""
        echo "  [3.3] Cross-dataset Classification: LN-INT -> LN-EXT"

        # Baseline: copy LN-INT model to LN-EXT and test
        mkdir -p runs/biomedclip/cls_baseline_$seed/LN-EXT/train
        cp runs/biomedclip/cls_baseline_$seed/LN-INT/train/best_model.pth \
            runs/biomedclip/cls_baseline_$seed/LN-EXT/train/best_model.pth
        python -m src.models.biomedclip.classification \
            --dataset LN-EXT \
            --epochs $SUPERVISED_EPOCHS \
            --seed $seed \
            --exp biomedclip/cls_baseline_$seed \
            --test

        # MONA: copy LN-INT model to LN-EXT and test
        mkdir -p runs/biomedclip/cls_mona_${MONA_VARIANT}_$seed/LN-EXT/train
        cp runs/biomedclip/cls_mona_${MONA_VARIANT}_$seed/LN-INT/train/best_model.pth \
            runs/biomedclip/cls_mona_${MONA_VARIANT}_$seed/LN-EXT/train/best_model.pth
        python -m src.models.biomedclip.classification \
            --dataset LN-EXT \
            --mona_weights $MONA_FINETUNED_PATH \
            --mona_variant $MONA_VARIANT \
            --epochs $SUPERVISED_EPOCHS \
            --seed $seed \
            --exp biomedclip/cls_mona_${MONA_VARIANT}_$seed \
            --test

        # =========================================================================
        # Step 4: Supervised Segmentation
        # =========================================================================
        echo ""
        echo "[Step 4] Supervised Segmentation"
        echo ""

        # Without MONA (baseline)
        for dataset in "${SEG_DATASETS[@]}"; do
            echo "  [4.1] Segmentation on $dataset - Baseline"
            python -m src.models.biomedclip.segmentation \
                --dataset $dataset \
                --epochs $SUPERVISED_EPOCHS \
                --seed $seed \
                --exp biomedclip/seg_baseline_$seed
        done

        # With MONA fine-tuned weights
        for dataset in "${SEG_DATASETS[@]}"; do
            echo "  [4.2] Segmentation on $dataset - MONA ${MONA_VARIANT}"
            python -m src.models.biomedclip.segmentation \
                --dataset $dataset \
                --mona_weights $MONA_FINETUNED_PATH \
                --mona_variant $MONA_VARIANT \
                --epochs $SUPERVISED_EPOCHS \
                --seed $seed \
                --exp biomedclip/seg_mona_${MONA_VARIANT}_$seed
        done

        # Cross-dataset segmentation evaluation: LN-INT -> LN-EXT
        echo ""
        echo "  [4.3] Cross-dataset Segmentation: LN-INT -> LN-EXT"

        # Baseline: copy LN-INT model to LN-EXT and test
        mkdir -p runs/biomedclip/seg_baseline_$seed/LN-EXT/train
        cp runs/biomedclip/seg_baseline_$seed/LN-INT/train/best_model.pth \
            runs/biomedclip/seg_baseline_$seed/LN-EXT/train/best_model.pth
        python -m src.models.biomedclip.segmentation \
            --dataset LN-EXT \
            --epochs $SUPERVISED_EPOCHS \
            --seed $seed \
            --exp biomedclip/seg_baseline_$seed \
            --test

        # MONA: copy LN-INT model to LN-EXT and test
        mkdir -p runs/biomedclip/seg_mona_${MONA_VARIANT}_$seed/LN-EXT/train
        cp runs/biomedclip/seg_mona_${MONA_VARIANT}_$seed/LN-INT/train/best_model.pth \
            runs/biomedclip/seg_mona_${MONA_VARIANT}_$seed/LN-EXT/train/best_model.pth
        python -m src.models.biomedclip.segmentation \
            --dataset LN-EXT \
            --mona_weights $MONA_FINETUNED_PATH \
            --mona_variant $MONA_VARIANT \
            --epochs $SUPERVISED_EPOCHS \
            --seed $seed \
            --exp biomedclip/seg_mona_${MONA_VARIANT}_$seed \
            --test

        # =========================================================================
        # Step 5: Image-Text Retrieval
        # =========================================================================
        echo ""
        echo "[Step 5] Image-Text Retrieval"
        echo ""

        # Without MONA (baseline)
        echo "  [5.1] Retrieval - Baseline"
        python -m src.models.biomedclip.retrieval \
            --seed $seed \
            --exp biomedclip/retrieval_baseline_$seed

        # With MONA fine-tuned weights
        echo "  [5.2] Retrieval - MONA ${MONA_VARIANT}"
        python -m src.models.biomedclip.retrieval \
            --mona_weights $MONA_FINETUNED_PATH \
            --mona_variant $MONA_VARIANT \
            --seed $seed \
            --exp biomedclip/retrieval_mona_${MONA_VARIANT}_$seed

        echo ""
        echo "  ✓ Run $seed completed"
    done

    echo ""
    echo "========================================================================"
    echo "BiomedCLIP Experiment Summary"
    echo "========================================================================"
    echo ""
    echo "Completed experiments with ${#SEEDS[@]} random seeds:"
    echo ""
    echo "  1. Fine-tuning with Hybrid MONA adapter"
    echo "     - Variant: ${MONA_VARIANT}"
    echo "     - Epochs: ${FINETUNE_EPOCHS}"
    echo ""
    echo "  2. Zero-shot classification"
    echo "     - Datasets: ${ZS_DATASETS[*]}"
    echo "     - Methods: Baseline, MONA"
    echo ""
    echo "  3. Supervised classification"
    echo "     - Datasets: ${CLS_DATASETS[*]}"
    echo "     - Cross-dataset: LN-INT -> LN-EXT"
    echo "     - Methods: Baseline, MONA"
    echo "     - Epochs: ${SUPERVISED_EPOCHS}"
    echo ""
    echo "  4. Supervised segmentation"
    echo "     - Datasets: ${SEG_DATASETS[*]}"
    echo "     - Cross-dataset: LN-INT -> LN-EXT"
    echo "     - Methods: Baseline, MONA"
    echo "     - Epochs: ${SUPERVISED_EPOCHS}"
    echo ""
    echo "  5. Image-text retrieval (ROCO-V2)"
    echo "     - Methods: Baseline, MONA"
    echo ""
    echo "Results saved to: runs/biomedclip/"
    echo "Logs saved to: logs/biomedclip.log"
    echo ""

} 2>&1 | tee -a logs/biomedclip.log
