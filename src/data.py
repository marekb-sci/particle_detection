# -*- coding: utf-8 -*-

import torch
from torchvision import transforms

def get_augmentation_tv(aug_list):

    transform = transforms.Compose([
        transforms.__getattribute__(aug[0])(**aug[1]) for aug in aug_list
        ])
    return transform

AUG_LIST_FLIPS = [('RandomHorizontalFlip', {}), ('RandomVerticalFlip', {})]
