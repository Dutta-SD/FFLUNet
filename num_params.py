from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetAttentionDynamicShift import (
    FFLUNetAttentionDynamicShift,
)

# TODO Convert to Script
model = FFLUNetAttentionDynamicShift(4, 3)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("Params", count_parameters(model))
