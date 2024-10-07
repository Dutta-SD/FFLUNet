import os

from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetAttentionDynamicShift import (
    FFLUNetAttentionDynamicShift,
)
from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetDynamicShift4Layers import (
    FFLUNetDynamicWindowShift4Layers,
)

os.environ["nnUNet_raw"] = "data/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "data/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "data/nnUNet_results"

import json
from calflops import calculate_flops
import torch
import time
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetDynamicShift import (
    FFLUNetDynamicWindowShift,
)
from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetDynamicShift12M import (
    FFLUNetDynamicWindowShift12M,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
import numpy as np

CONFIG_NAME = "3d_fullres"
MODEL_NAME = "FFLUNETATTENTION"


with open("data/nnUNet_preprocessed/Dataset980_BraTS2023/nnUNetPlans.json") as fp:
    plans = json.load(fp)
    pm = PlansManager(plans)

with open("data/nnUNet_preprocessed/Dataset980_BraTS2023/dataset.json") as fp:
    djson = json.load(fp)


if MODEL_NAME == "NNUNET":
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
elif MODEL_NAME == "FFLUNET12M":
    model = FFLUNetDynamicWindowShift12M(4, 3)
elif MODEL_NAME == "FFLUNET":
    model = FFLUNetDynamicWindowShift(4, 3)
elif MODEL_NAME == "FFLUNET4LAYERS":
    model = FFLUNetDynamicWindowShift4Layers(4, 3)
elif MODEL_NAME == "FFLUNETATTENTION":
    model = FFLUNetAttentionDynamicShift(4, 3)

batch_size = 1
input_shape = (batch_size, 4, 128, 128, 128)
flops, macs, params = calculate_flops(
    model=model,
    input_shape=input_shape,
    print_detailed=False,
    print_results=False,
    output_as_string=True,
    output_precision=4,
)
print(f"{model.__class__.__name__} FLOPs:{flops}, MACs:{macs}, Params:{params}")

# Assuming input_shape and model are already defined
times = []

for _ in range(10):
    start = time.time()
    with torch.no_grad():
        t = torch.rand(*input_shape)
        model(t)
    end = time.time()

    times.append(end - start)

average_time = np.mean(times)
std_deviation = np.std(times)

print(
    f"{model.__class__.__name__} Prediction time: {average_time:.3f} ± {std_deviation:.3f} s"
)
