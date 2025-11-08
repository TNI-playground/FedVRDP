'''
Implementation of Fang: Local model poisoning attacks to Byzantine-robust federated learning. 2020 
https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning
Fang is designed to break Krum defense but can be used to attack other defense methods
The fang_krum is designed for krum and bulyan
The fang_tr_mean is designed for both trimmed-mean and mediam

'''
from __future__ import print_function
import argparse, os, sys, csv, shutil, time, random, operator, pickle, ast, math, json
import numpy as np
import pandas as pd
from torch.optim import Optimizer
import torch.nn.functional as F
import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.multiprocessing as mp

##### code of fltrust ####
# def trim_attack(v, net, lr, f):
#     # local model poisoning attack against Trimmed-mean
#     vi_shape = v[0].shape
#     v_tran = nd.concat(*v, dim=1)
#     maximum_dim = nd.max(v_tran, axis=1).reshape(vi_shape)
#     minimum_dim = nd.min(v_tran, axis=1).reshape(vi_shape)
#     direction = nd.sign(nd.sum(nd.concat(*v, dim=1), axis=-1, keepdims=True))
#     directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim
#     # let the malicious clients (first f clients) perform the attack
#     for i in range(f):
#         random_12 = 1. + nd.random.uniform(shape=vi_shape)
#         v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
#     return v 

##### code ndss 21
def fang_trmean_median_white(all_updates, args):
    '''
    attackers know all the benign updates: all_updates[n_attackers:]
    the attacker uses all the benign updates including its or not?: 
    fltrust use all but ndss21 use benign's only (and their parameters b are different)
    '''
    
    # flatten model parameters
    all_benign_updates_flatten=[]
    for update in all_updates[args.num_attackers:]:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_benign_updates_flatten = update[None, :] if not len(all_benign_updates_flatten) else torch.cat((all_benign_updates_flatten, update[None, :]), 0)
    
    deviation = torch.sign(torch.mean(all_benign_updates_flatten, 0))

    b = 2
    max_vector = torch.max(all_benign_updates_flatten, 0)[0]
    min_vector = torch.min(all_benign_updates_flatten, 0)[0]

# #### fltrust's code: remember change to that fltrust use benign model of all users rather than only benign users' models
#     flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
#     idx = []
#     s = 0
#     for p in flattened:
#         d = p.shape[0]
#         idx.append((s, s + d))
#         s += d
#     directed_dim = (deviation > 0) * min_vector + (deviation < 0) * max_vector
#     for i in range(n_attackers):
#         random_12 = torch.from_numpy(1 + np.random.uniform(0, 1, deviation.shape)).type(torch.FloatTensor).cuda(args.device)
#         v = directed_dim * ((deviation * directed_dim > 0) / random_12 + (deviation * directed_dim < 0) * random_12)
#         all_updates[i] = {k: v[s:d].reshape(all_updates[-1][k].shape) for k, (s, d) in zip(all_updates[-1].keys(), idx)}


    max_ = (max_vector > 0).type(torch.FloatTensor).cuda().to(args.device)
    min_ = (min_vector < 0).type(torch.FloatTensor).cuda().to(args.device)

    max_[max_ == 1] = b
    max_[max_ == 0] = 1 / b
    min_[min_ == 1] = b
    min_[min_ == 0] = 1 / b

    max_range = torch.cat((max_vector[:, None], (max_vector * max_)[:, None]), dim=1)
    min_range = torch.cat(((min_vector * min_)[:, None], min_vector[:, None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), args.num_attackers])).type(torch.FloatTensor).cuda(args.device)

    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

    mal_vec = (torch.stack([(deviation < 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda(args.device) * max_rand + torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda(args.device) * min_rand).T
    
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    for i in range(args.num_attackers):
        all_updates[i] = {k: mal_vec[i,:][s:d].reshape(all_updates[-1][k].shape) for k, (s, d) in zip(all_updates[-1].keys(), idx)}

    return all_updates

def fang_trmean_median_gray(all_updates, args):
    '''
    attackers do not know the benign models, and use their own models
    '''
    # flatten attacker's model parameters only
    all_benign_updates_flatten=[]
    for update in all_updates[:args.num_attackers]:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_benign_updates_flatten = update[None, :] if not len(all_benign_updates_flatten) else torch.cat((all_benign_updates_flatten, update[None, :]), 0)

    model_re = torch.mean(all_benign_updates_flatten, 0)
    model_std = torch.std(all_benign_updates_flatten, 0)
    deviation = torch.sign(model_re)
    
    max_vector_low = model_re + 3 * model_std 
    max_vector_hig = model_re + 4 * model_std
    min_vector_low = model_re - 4 * model_std
    min_vector_hig = model_re - 3 * model_std

    max_range = torch.cat((max_vector_low[:,None], max_vector_hig[:,None]), dim=1)
    min_range = torch.cat((min_vector_low[:,None], min_vector_hig[:,None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), args.num_attackers])).type(torch.FloatTensor).cuda(args.device)

    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

    mal_vec = (torch.stack([(deviation < 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda(args.device) * max_rand + torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda(args.device) * min_rand).T

    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    # mal_models=[]
    for i in range(args.num_attackers):
        all_updates[i] = {k: mal_vec[i,:][s:d].reshape(all_updates[-1][k].shape) for k, (s, d) in zip(all_updates[-1].keys(), idx)}

    return all_updates

def multi_krum_(updates, n_attackers, multi_k):
    '''
    multi_k = False for fang's attack
    '''
    num_users = len(updates)
    
    candidates = []
    candidate_indices = []
    remaining_updates = updates
    all_indices = np.arange(num_users)
    while len(remaining_updates) > 2 * n_attackers + 2:
        distances = []
        for update in remaining_updates:
            distance = torch.norm((remaining_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        distances = torch.sort(distances, dim=1)[0]
        
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]
        
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
    
        if not multi_k:
            break
    aggregate = torch.mean(candidates, dim=0)
    return aggregate, np.array(candidate_indices)

def compute_lambda(all_updates, model_re, n_attackers):

    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1)
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    distances[distances == 0] = 10000
    distances = torch.sort(distances, dim=1)[0]
    scores = torch.sum(distances[:, :n_benign - 2 - n_attackers], dim=1)
    min_score = torch.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0])
    max_wre_dist = torch.max(torch.norm((all_updates - model_re), dim=1)) / (torch.sqrt(torch.Tensor([d]))[0])

    return (term_1 + max_wre_dist)

def fang_krum_bulyan_gray(all_updates, args):
    '''
    attackers do not know the benign models, and use their own models
    n_attacker_ = max(1, n_attacker**2//nusers): check why ndss2021 use this for gray box attack
    '''
    n_attackers = max(1, args.num_attackers**2//args.num_selected_users)
    # flatten attacker's model parameters only
    all_benign_updates_flatten=[]
    for update in all_updates[:args.num_attackers]:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_benign_updates_flatten = update[None, :] if not len(all_benign_updates_flatten) else torch.cat((all_benign_updates_flatten, update[None, :]), 0)

    model_re = torch.mean(all_benign_updates_flatten, 0)
    deviation = torch.sign(model_re)

    lamda = compute_lambda(all_benign_updates_flatten, model_re, n_attackers)
    threshold = 1e-5

    mal_updates = []
    while lamda > threshold:
        mal_update = (- lamda * deviation)

        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_benign_updates_flatten), 0)

        if args.defend == 'bulyan':
            _, krum_candidate = multi_krum_(mal_updates, n_attackers, multi_k=True)
        else:
            _, krum_candidate = multi_krum_(mal_updates, n_attackers, multi_k=False)

        # print(krum_candidate)
        if len(krum_candidate) == 1:
            if krum_candidate < n_attackers:
                break
        else:
            if np.sum(krum_candidate < n_attackers) >= (n_attackers // 2):
                break

        lamda *= 0.5

    if not len(mal_updates):
        print(lamda, threshold)
        mal_update = (model_re - lamda * deviation)
    # print(len({mal_update}*args.num_attackers))
    # exit('end')
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    mal_update = {k: mal_update[s:d].reshape(all_updates[-1][k].shape) for k, (s, d) in zip(all_updates[-1].keys(), idx)}
    for i in range(args.num_attackers):
        all_updates[i] = mal_update
    return all_updates