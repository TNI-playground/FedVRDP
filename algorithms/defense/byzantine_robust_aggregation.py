'''
multi-krum, 
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

def multi_krum(all_updates, n_attackers=10, multi_k=False):
    num_users = len(all_updates)
    # flatten model parameters
    all_updates_flatten=[]
    for update in all_updates:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_updates_flatten = update[None, :] if not len(all_updates_flatten) else torch.cat((all_updates_flatten, update[None, :]), 0)
        
    candidates = []
    candidate_indices = []
    remaining_updates = all_updates_flatten
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
    aggregate = torch.mean(candidates, dim=0) # mean for multi-krum, this line doesn't influence krum
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    aggregate_model = {k: aggregate[s:d].reshape(all_updates[0][k].shape) for k, (s, d) in zip(all_updates[0].keys(), idx)}

    return aggregate_model, np.array(candidate_indices)

def bulyan(all_updates, n_attackers=10):
    num_users = len(all_updates)
    # flatten model parameters
    all_updates_flatten=[]
    for update in all_updates:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_updates_flatten = update[None, :] if not len(all_updates_flatten) else torch.cat((all_updates_flatten, update[None, :]), 0)
        
    bulyan_cluster = []
    candidate_indices = []
    remaining_updates = all_updates_flatten
    all_indices = np.arange(num_users)

    while len(bulyan_cluster) < (num_users - 2 * n_attackers):
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        # print(distances)
        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]
        if not len(indices):
            break
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        bulyan_cluster = remaining_updates[indices[0]][None, :] if not len(bulyan_cluster) else torch.cat((bulyan_cluster, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

    # print('dim of bulyan cluster ', bulyan_cluster.shape)

    n, d = bulyan_cluster.shape
    param_med = torch.median(bulyan_cluster, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
    sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]
    
    # aggregate = torch.mean(sorted_params[:n - 2 * n_attackers], dim=0)
    aggregate = torch.mean(sorted_params[n_attackers:-n_attackers], 0) # trimmed mean

    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    aggregate_model = {k: aggregate[s:d].reshape(all_updates[0][k].shape) for k, (s, d) in zip(all_updates[0].keys(), idx)}

    return aggregate_model, np.array(candidate_indices)

def tr_mean(all_updates, n_attackers=10):
    all_updates_flatten=[]
    for update in all_updates:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_updates_flatten = update[None, :] if not len(all_updates_flatten) else torch.cat((all_updates_flatten, update[None, :]), 0)
        
    sorted_updates = torch.sort(all_updates_flatten, 0)[0]
    aggregate = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates,0)
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    aggregate_model = {k: aggregate[s:d].reshape(all_updates[0][k].shape) for k, (s, d) in zip(all_updates[0].keys(), idx)}

    return aggregate_model

def median(all_updates, n_attackers=10):
    all_updates_flatten=[]
    for update in all_updates:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_updates_flatten = update[None, :] if not len(all_updates_flatten) else torch.cat((all_updates_flatten, update[None, :]), 0)
        
    sorted_updates = torch.sort(all_updates_flatten, 0)[0]
    aggregate = torch.median(sorted_updates[n_attackers:-n_attackers], 0)[0] if n_attackers else torch.mean(sorted_updates,0)
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    aggregate_model = {k: aggregate[s:d].reshape(all_updates[0][k].shape) for k, (s, d) in zip(all_updates[0].keys(), idx)}

    return aggregate_model

def coordinate_median(all_updates):
    all_updates_flatten=[]
    for update in all_updates:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_updates_flatten = update[None, :] if not len(all_updates_flatten) else torch.cat((all_updates_flatten, update[None, :]), 0)
        
    sorted_updates = torch.sort(all_updates_flatten, 0)[0]
    aggregate = torch.median(sorted_updates, 0)[0]
    # reshape to model
    flattened = [torch.flatten(all_updates[0][k]) for k in all_updates[0].keys()]
    idx = []
    s = 0
    for p in flattened:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    aggregate_model = {k: aggregate[s:d].reshape(all_updates[0][k].shape) for k, (s, d) in zip(all_updates[0].keys(), idx)}

    return aggregate_model