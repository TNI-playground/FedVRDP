import random
import torch
import logging
from collections import OrderedDict
import math
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

################################# constrain #################################
def model_dist_norm_var(model, target_params_variables, norm=2):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.cuda.FloatTensor(size).fill_(0)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
        layer - target_params_variables[name]).view(-1)
        size += layer.view(-1).shape[0]

    return torch.norm(sum_var, norm)

################################# scale #################################
def scale(model, target_model):
    ### We scale data according to formula: L = 100*X-99*G = G + (100*X- 100*G).
    clip_rate = (helper.params['scale_weights'] / current_number_of_adversaries)
    for key, value in model.state_dict().items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and key == 'decoder.weight' or '__'in key:
            continue
        target_value = target_model.state_dict()[key]
        new_value = target_value + (value - target_value) * clip_rate

        model.state_dict()[key].copy_(new_value)
    return model
