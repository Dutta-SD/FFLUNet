import os
from time import time
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

TRAINED_MODEL_ROOT = "Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_lowres"
CHECKPOINT_NAME = "checkpoint_final.pth"
DATASET_ID = "080"
DATASET_NAME = "Brats2020"
TEST_INPUT_ROOT_DIR = f"Dataset{DATASET_ID}_{DATASET_NAME}/imagesTs"


if __name__ == "__main__":
    CURR_TIME = int(time.time())
    OUTPUT_ROOT_DIR = f"Dataset{DATASET_ID}_{DATASET_NAME}/pred/{CURR_TIME}"
    os.makedirs(TEST_INPUT_ROOT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ROOT_DIR)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, TRAINED_MODEL_ROOT),
        use_folds=None,
        checkpoint_name=CHECKPOINT_NAME,
    )

    predictor.predict_from_files(
        join(nnUNet_raw, TEST_INPUT_ROOT_DIR),
        join(nnUNet_raw, OUTPUT_ROOT_DIR),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=3,
        num_processes_segmentation_export=3,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )
