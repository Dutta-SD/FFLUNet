# ***********************
# Set NNUNET variables before running this code
#
# ************************
from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetAttentionDynamicShift import (
    FFLUNetAttentionDynamicShift,
)
from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetDynamicShift4Layers import (
    FFLUNetDynamicWindowShift4Layers,
)
from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetDynamicShift import (
    FFLUNetDynamicWindowShift,
)
from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetDynamicShift12M import (
    FFLUNetDynamicWindowShift12M,
)
import json
from calflops import calculate_flops
import torch
import time
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
import numpy as np
import torch
from monai.inferers import sliding_window_inference

CONFIG_NAME = "3d_fullres"
INPUT_SHAPE = (1, 4, 128, 128, 128)
I_CH = 4
O_CH = 3

with open("data/nnUNet_preprocessed/Dataset980_BraTS2023/nnUNetPlans.json") as fp:
    plans = json.load(fp)
    pm = PlansManager(plans)

with open("data/nnUNet_preprocessed/Dataset980_BraTS2023/dataset.json") as fp:
    djson = json.load(fp)


def get_avg_std(val_list):
    average_time = np.mean(val_list)
    std_deviation = np.std(val_list)
    return average_time, std_deviation


def benchmark(model, num_simulation_runs):
    # Flops, MACs
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=INPUT_SHAPE,
        print_detailed=False,
        print_results=False,
        output_as_string=True,
        output_precision=4,
    )
    print(
        f"[{model.__class__.__name__}]: (FLOPs:{flops}, MACs:{macs}, #Params:{params})"
    )

    # Inference time
    cpu_times, gpu_times = [], []

    roi_size = (128, 128, 128)
    sw_batch_size = 4
    overlap = 0.5

    model = model.cpu()

    for i in range(num_simulation_runs):
        print(i)
        image = torch.rand(1, 4, 240, 240, 155).cpu()

        # CPU Inference time
        s = time.time()
        with torch.no_grad():
            sliding_window_inference(
                image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )
        e = time.time()
        cpu_times.append(e - s)

    model = model.cuda()

    print(
        "{} CPU Inference: {:.3f} ± {:.3f} s".format(
            model.__class__.__name__, *get_avg_std(cpu_times)
        )
    )

    for i in range(num_simulation_runs):
        print(i)
        image = torch.rand(1, 4, 240, 240, 155).cuda()

        # GPU Inference time
        torch.cuda.synchronize()
        s = time.time()
        with torch.no_grad():
            sliding_window_inference(
                image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )
        torch.cuda.synchronize()
        e = time.time()
        gpu_times.append(e - s)
    print(
        "{} GPU Inference: {:.3f} ± {:.3f} s".format(
            model.__class__.__name__, *get_avg_std(gpu_times)
        )
    )


def get_model(model_name):
    model = None
    if model_name == "NNUNET":
        trainer = nnUNetTrainer(
            plans=plans,
            configuration=CONFIG_NAME,
            fold=1,
            dataset_json=djson,
        )
        model = trainer.build_network_architecture(
            plans_manager=pm,
            dataset_json=djson,
            configuration_manager=pm.get_configuration(CONFIG_NAME),
            num_input_channels=4,
            enable_deep_supervision=True,
        )
    elif model_name == "FFLUNET12M":
        model = FFLUNetDynamicWindowShift12M(4, 3)
    elif model_name == "FFLUNET":
        model = FFLUNetDynamicWindowShift(4, 3)
    elif model_name == "FFLUNET4LAYERS":
        model = FFLUNetDynamicWindowShift4Layers(4, 3)
    elif model_name == "FFLUNETATTENTION":
        model = FFLUNetAttentionDynamicShift(4, 3)
    return model


if __name__ == "__main__":
    all_model_names = [
        "NNUNET",
        "FFLUNET",
        "FFLUNET4LAYERS",
        "FFLUNETATTENTION",
        "FFLUNET12M",
    ]

    for model_name in all_model_names:
        model = get_model(model_name)
        benchmark(model, 5)
