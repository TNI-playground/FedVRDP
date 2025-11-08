#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from tkinter.messagebox import NO
import copy
from skimage import img_as_ubyte

def dba_poison(images, labels, args, trigger_cnt=-1, evaluation=False):
    new_images = images
    new_targets = labels

    for index in range(0, len(images)):
        if evaluation: # poison all data when testing
            new_images[index] = add_pixel_pattern(args, images[index], trigger_cnt)

        else: # poison part of data when training
            if index < args.poisoning_per_batch:
                new_targets[index] = args.targeted_poison_label
                new_images[index] = add_pixel_pattern(args, images[index], trigger_cnt)
            else:
                new_images[index] = images[index]
                new_targets[index]= labels[index]

    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad_(False)
    return new_images, new_targets

def add_pixel_pattern(args, ori_image, adversarial_index):
    image = copy.deepcopy(ori_image)
    poison_patterns = []
    if adversarial_index == -1:
        for i in range(0, args.trigger_num):
            poison_patterns = poison_patterns + getattr(args, 'poison_pattern')[i]
    else :
        poison_patterns = getattr(args, 'poison_pattern')[adversarial_index]
    if args.dataset == 'cifar' or args.dataset == 'tiny_imagenet':
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
            image[1][pos[0]][pos[1]] = 1
            image[2][pos[0]][pos[1]] = 1
    elif args.dataset == 'mnist':
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
    return image