# -*- coding: utf-8 -*-

import torchmetrics
import torch
import numpy as np
from PIL import Image


class DummyMetric(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("value", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, value, weight=1.):

        self.value += value*weight
        self.weight += weight

    def compute(self):
        return self.value.float() / self.weight

class ImageLoader:
    def __init__(self, depth=8):
        self.max_val = 2**depth -1

    def __call__(self, img_path):
        img_pil = Image.open(img_path)
        tensor = torch.Tensor(np.array(img_pil) / self.max_val)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor