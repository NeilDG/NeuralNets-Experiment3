"""
Customized transforms using kornia for faster data augmentation

@author: delgallegon
"""

import torch
import torch.nn as nn
import kornia
import numpy as np

class CycleGANTransform(nn.Module):

    def __init__(self, opts):
        super(CycleGANTransform, self).__init__()

        self.patch_size = (opts.patch_size, opts.patch_size)
        self.grid_size = (opts.patch_size / 16, opts.patch_size / 16)

        self.transform_op = kornia.augmentation.PatchSequential(
            kornia.augmentation.RandomVerticalFlip(p=0.5),
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
        grid_size=(self.grid_size),
        patchwise_apply=False,
        same_on_batch=True)

        self.patch_extract_op = kornia.contrib.ExtractTensorPatches(window_size=self.patch_size, stride=self.patch_size)


    def forward(self, x):
        out_tensor = self.transform_op(x)
        out_tensor = self.patch_extract_op(out_tensor)
        out_tensor = torch.flatten(out_tensor, 0, 1)
        # print("Shape: ", np.shape(out_tensor))
        return out_tensor