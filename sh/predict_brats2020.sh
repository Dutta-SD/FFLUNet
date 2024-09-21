#!/bin/bash
#SBATCH --job-name=brats2020_fflunet_predict
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=gpupart_1hour
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=2:58:00
#SBATCH --output=logs/%u_%x_%j.out

conda activate gpu
nvidia-smi
export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="data/nnUNet_results"

nnUNetv2_predict_brats2020 -tr Dataset080_BraTS2020/nnUNetTrainer_FFLUNetAttentionDynamicShift__nnUNetPlans__3d_fullres 
