#!/bin/bash
#SBATCH --job-name=FFLUNet     # Job name
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --mem=32gb                   # Job memory request
#SBATCH --partition=gpupart_48hour       # GPU Partition
#SBATCH --gpus-per-node=1            #number of GPUS
#SBATCH --time=0-22:00:00            # Time limit hrs:min:sec
#SBATCH --output=logs/job_%j.log    # Standard output and error log

#module load anaconda3
conda activate gpu
export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="data/nnUNet_results"

# nnUNetv2_plan_and_preprocess -d 080 --verify_dataset_integrity
nnUNetv2_train 080 3d_fullres 1 -tr nnUNetTrainer_FFLUNetAttentionDynamicShift

