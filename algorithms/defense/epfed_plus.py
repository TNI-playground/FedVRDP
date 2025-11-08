'''
Efficient and Private Federated Learning with Sparsely DP in Federated Learning
'''
import torch
from torch import nn
import copy
import numpy as np

from ..privacy.dp_mechanism import cal_sensitivity, Laplace, Gaussian_Simple
from ..privacy.dp_compress import private_com, cpsgd

def topk(vector, args):
    '''
    return the mask for topk of vector
    '''
    # on a gpu, sorting is faster than pytorch's topk method
    #topkIndices = torch.sort(vec**2)[1][-k:]
    # however, torch.topk is more space efficient

    # topk on cuda returns what looks like uninitialized memory if
    # vals has nan values in it
    # saving to a zero-initialized output array instead of using the
    # output of topk appears to solve this problem
    # compressed = [torch.flatten(model[k]) for k in model.keys()]
    # idx = []
    # s = 0
    # for p in compressed:
    #     d = p.shape[0]
    #     idx.append((s, s + d))
    #     s += d
    # flat = torch.cat(compressed).view(1, -1)
    # flat = flat.flatten()

    k_dim = int(args.g_com_p * args.dim)
    
    mask = torch.zeros_like(vector)
    # flat_abs = abs(flat)
    _, indices = torch.topk(vector**2, k_dim)
    # generate a mask, set topk as 1, otherwise 0
    mask[indices] = 1
    # mask = {k: mask[s:d].reshape(model[k].shape) for k, (s, d) in zip(model.keys(), idx)}
    return mask, mask*vector

def model_topk(vector, args):
    '''
    return the mask for topk of model
    '''
    k_dim = int(args.com_p * args.dim)

    # vector = parameters_dict_to_vector_flt(model)
    
    mask = torch.zeros_like(vector)
    # flat_abs = abs(flat)
    _, indices = torch.topk(vector**2, k_dim)
    # generate a mask, set topk as 1, otherwise 0
    mask[indices] = 1
    # mask = {k: mask[s:d].reshape(model[k].shape) for k, (s, d) in zip(model.keys(), idx)}
    return mask

def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def no_defence_balance_(ave_model_update, global_model, local_updates):
    '''
    ave_model_update is a vector; others are dicts; 
    change the corresponding parts in fang and flame
    '''
    pointer = 0
    model_update = copy.deepcopy(global_model)

    for key, param in model_update.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        num_param = param.numel()
        param.data = ave_model_update[pointer:pointer + num_param].view_as(param).data
        pointer += num_param

    for var in global_model:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_model[var] = local_updates[0][var]
            continue
        global_model[var] += model_update[var]
    return global_model

def vector_to_net_dict(vec: torch.Tensor, net_dict) -> None:
    r"""Convert one vector to the net parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict

################################# DP ######################################
def add_noise(net, args):
    sensitivity = cal_sensitivity(args.lr, args.dp_clip, len(idxs))
    if args.dp_mechanism == 'Laplace':
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = Laplace(epsilon=args.dp_epsilon, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(args.device)
                v += noise
    elif args.dp_mechanism == 'Gaussian':
        with torch.no_grad():
            for k, v in net.named_parameters():
                noise = Gaussian_Simple(epsilon=args.dp_epsilon, delta=args.dp_delta, sensitivity=sensitivity, size=v.shape)
                noise = torch.from_numpy(noise).to(args.device)
                v += noise

def gaussian_mechanism(model, args):
    '''
    gaussian mechanism on model update
    '''
    model = {k: model[k] + torch.normal(0, args.sigma, model[k].size()).to(args.device)
        for k in model.keys()}
    return model

def epfed_plus(local_updates, global_model, args, momentum=None, momentum_tilde=None, global_mask=None):
    ###########################
    ########## local ##########
    ###########################
    flat_local_updates = []
    for param in local_updates:
        flat_param = parameters_dict_to_vector_flt(param)
        # compute topk value
        if global_mask is None:
            global_mask = torch.zeros_like(flat_param)
            random_mask = torch.randn_like(flat_param)
            _, indices = torch.topk(random_mask**2, int(args.com_p * args.dim))
            global_mask[indices] = 1
        # flat_param = torch.mul(flat_param, global_mask)
        flat_param = flat_param * global_mask

        # Clip
        delta_norm = torch.norm(flat_param)
        threshold = delta_norm / args.clip
        if threshold > 1.0:
            flat_param = flat_param / threshold

        # Add DP noise
        if args.use_dp:
            args.sigma = args.noise_multiplier * args.clip / np.sqrt(args.num_selected_users)
            dp_noise = torch.normal(0, args.sigma, flat_param.shape).to(args.device) * global_mask
            flat_param = flat_param + dp_noise

        flat_local_updates.append(flat_param)
    print("sigma ", args.sigma)
    ###########################
    ########## global #########
    ###########################
    # flatten models
    local_updates_vector = torch.stack(flat_local_updates, dim=0)
    # local_updates_vector = []
    # for param in flat_local_updates:
    #     local_updates_vector = param[None, :] if len(local_updates_vector) == 0 \
    #                                           else torch.cat((local_updates_vector, \
    #                                                           param[None, :]), 0)

    # aggregate local model updates
    avg_local_update = torch.mean(local_updates_vector, 0)

    # momentum update on the server side # remember to add this to other algorithms
    if momentum is None:
        momentum = torch.zeros_like(avg_local_update)
    if momentum_tilde is None:
        momentum_tilde = torch.zeros_like(avg_local_update)
    
    if args.use_momentum:
        print('use momentum')
        avg_local_update = args.momentum_beta * momentum + avg_local_update
        momentum = avg_local_update.clone().detach()
        args.momentum_beta = args.momentum_beta * args.decay_weight
    elif args.use_boost_momentum:
        print('use Boost Momentum')
        momentum = avg_local_update.clone().detach()
        beta = args.boost_momentum_alpha * args.boost_momentum_eta_g**2
        avg_local_update = beta * avg_local_update + (1-beta) * momentum_tilde + (1-beta) * (avg_local_update - momentum)
        momentum_tilde = avg_local_update.clone().detach()
        args.boost_momentum_eta_g *= args.boost_momentum_eta_g_decay
    elif args.use_adam:
        print('use Adam')
        momentum = args.adam_beta_1 * momentum + (1-args.adam_beta_1) * avg_local_update
        momentum_ = momentum / (1-args.adam_beta_1)
        momentum_tilde = args.adam_beta_2 * momentum_tilde + (1-args.adam_beta_2) * avg_local_update**2
        momentum_tilde_ = momentum_tilde / (1-args.adam_beta_2)
        momentum = momentum.clone().detach()
        momentum_tilde = momentum_tilde.clone().detach()
    elif args.use_adagrad:
        print('use AdaGrad')
        momentum = args.adagrad_beta_1 * momentum + (1-args.adagrad_beta_1) * avg_local_update
        momentum_tilde = momentum_tilde + avg_local_update**2
        momentum = momentum.clone().detach()
        momentum_tilde = momentum_tilde.clone().detach()
    elif args.use_yogi:
        print('use Yogi')
        momentum = args.yogi_beta_1 * momentum + (1-args.yogi_beta_1) * avg_local_update
        momentum_tilde = momentum_tilde - (1-args.yogi_beta_2) * avg_local_update**2 * torch.sign(momentum_tilde - avg_local_update**2)
        momentum = momentum.clone().detach()
        momentum_tilde = momentum_tilde.clone().detach()
    elif args.use_lion:
        print("use Lion")
        momentum_local_update = args.lion_beta_1 * momentum + (1-args.lion_beta_1) * avg_local_update
        momentum = args.lion_beta_2 * momentum + (1-args.lion_beta_2) * avg_local_update

    # Clip
    # if args.use_momentum or args.use_boost_momentum:
    #     global_delta_norm = torch.norm(avg_local_update)
    #     threshold = global_delta_norm / args.global_clip
    #     if threshold > 1.0:
    #         avg_local_update = avg_local_update / threshold
    # elif args.use_adam or args.use_adagrad or args.use_yogi:
    #     momentum_norm = torch.norm(momentum)
    #     threshold_m = momentum_norm / args.global_clip
    #     if threshold_m > 1.0:
    #         momentum = momentum / threshold_m
    #     momentum_tilde_norm = torch.norm(momentum_tilde)
    #     threshold_mt = momentum_tilde_norm / args.global_clip
    #     if threshold_mt > 1.0:
    #         momentum_tilde = momentum_tilde / threshold_mt
    # elif args.use_lion:
    #     momentum_norm = torch.norm(momentum)
    #     threshold_m = momentum_norm / args.global_clip
    #     if threshold_m > 1.0:
    #         momentum = momentum / threshold_m

    flat_global_model = parameters_dict_to_vector_flt(global_model)
    # flat_global_model = flat_global_model + avg_local_update
    # flat_global_model = flat_global_model + args.eta_g * avg_local_update
    if args.use_momentum:
        flat_global_model = flat_global_model + args.momentum_eta_g * avg_local_update
    elif args.use_boost_momentum:
        flat_global_model = flat_global_model + args.boost_momentum_eta_g * avg_local_update
    elif args.use_adam:
        flat_global_model = flat_global_model + args.adam_eta_g * momentum_ / (torch.sqrt(momentum_tilde_)+10e-3)
    elif args.use_adagrad:
        # _, momentum = topk(momentum, args)
        # _, momentum_tilde = topk(momentum_tilde, args)
        flat_global_model = flat_global_model + args.adagrad_eta_g * momentum / (torch.sqrt(momentum_tilde)+10e-3)
    elif args.use_yogi:
        flat_global_model = flat_global_model + args.yogi_eta_g * momentum / (torch.sqrt(momentum_tilde)+10e-3)
    elif args.use_lion:
        flat_global_model = flat_global_model + args.lion_eta_g * (torch.sign(momentum_local_update)+ args.lion_lamda * flat_global_model)
    else:
        # _, sparsed_avg_local_update = topk(c, args)
        # flat_global_model = flat_global_model + sparsed_avg_local_update
        flat_global_model = flat_global_model + avg_local_update

    # _, sparsed_local_update = topk(compensated_local_update, args)
    # convert vector to net and update global model
    global_model = vector_to_net_dict(flat_global_model, global_model)

    # compute topk mask
    global_mask = model_topk(flat_global_model, args)

    return global_model, momentum, momentum_tilde, global_mask
    # return global_model, momentum, momentum2order, global_mask
    