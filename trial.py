from nnunetv2.training.nnUNetTrainer.variants.fflunet.nnUNetTrainer_FFLUNetDynamicShift12M import (
    FFLUNetDynamicWindowShift12M,
)


import torch


model = FFLUNetDynamicWindowShift12M(4, 3).cuda()
t = torch.rand(4, 4, 128, 128, 128).cuda()

z = model(t)

l = z.sum()
l.backward()