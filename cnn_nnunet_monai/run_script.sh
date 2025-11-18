#!/bin/bash

# Set nnU-Net environment paths
export nnUNet_raw="./workdir/nnUNet_raw"
export nnUNet_preprocessed="./workdir/nnUNet_preprocessed"
export nnUNet_results="./workdir/nnUNet_results"

# Calculate memory limits (70% of available RAM)
TOTAL_MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
LIMIT_MEM_KB=$((TOTAL_MEM_KB * 7 / 10))

ulimit -m "$LIMIT_MEM_KB"
ulimit -v "$LIMIT_MEM_KB"

echo "Total RAM: $((TOTAL_MEM_KB / 1024 / 1024)) GB"
echo "Setting memory limit to: $((LIMIT_MEM_KB / 1024 / 1024)) GB"
echo "Starting nnU-Net pipeline..."

# Preprocessing
nnUNetv2_plan_and_preprocess -d 108 -c 3d_fullres --verify_dataset_integrity

# Training (folds 0–4)
for FOLD in 0 1 2 3 4; do
    nnUNetv2_train Dataset108_picai 3d_fullres "$FOLD" \
        -tr nnUNetTrainerCELoss_1000epochs --npz
done
