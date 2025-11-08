#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from tkinter.messagebox import NO
import copy
from skimage import img_as_ubyte
import numpy as np
import torch

def edge_poison(images, labels, args, evaluation=False):
    new_images = images
    new_targets = labels

    for index in range(0, len(images)):
        if evaluation:
            if labels[index] == args.targeted_poison_label:
                new_images[index] = images[index, :] / 255
            else:
                new_images[index] = images[index]
        else:
            if index < (len(images) * args.poison_ratio):
                if labels[index] == args.targeted_poison_label:
                    new_images[index] = images[index, :] / 255
                    new_targets[index] = 1
                else:
                    new_images[index] = images[index]
                    new_targets[index]= labels[index]
            else:
                new_images[index] = images[index]
                new_targets[index]= labels[index]

    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad_(False)
    return new_images, new_targets
