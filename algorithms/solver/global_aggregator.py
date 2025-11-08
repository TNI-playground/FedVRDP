import torch
import random
import numpy as np
import copy
from utils.model_utils import model_clip
from ..privacy.dp_compress import private_com

def average(global_model, local_updates):
    '''
    simple average
    '''
    model_update = {k: local_updates[0][k] *0.0 for k in local_updates[0].keys()}
    for i in range(len(local_updates)):
        model_update = {k: model_update[k] +  local_updates[i][k] for k in global_model.keys()}
    global_model = {k: global_model[k] +  model_update[k]/ len(local_updates) for k in global_model.keys()}
    return global_model

def gossip_average(local_updates, idxs_users, W, W_c):
    '''
    gossip average
    idxs_users must be all users
    '''
    local_models = copy.deepcopy(local_updates)
    for i in idxs_users:
        # find the neighbor of i and its weight vector
        local_model = {k: local_updates[i][k] * W[i][i] for k in local_updates[i].keys()}
        neighbors_idx = np.nonzero(W_c[i])
        ############# gathering neighbor's data ##########
        for j in neighbors_idx:
            local_model = {k: local_model[k] +  local_updates[j][k] * W[i][j] for k in local_model.keys()}
        local_models[i] = local_model 
    return local_models

def aggregation_sampled_avg(local_models, groups, args):
    # sampled average
    id_set=[]
    idx = random.choice(list(groups[0]))
    id_set.append(idx)
    global_model = local_models[idx]
    for g in range(1, args.num_groups):
        idx = random.choice(list(groups[g]))
        id_set.append(idx)
        global_model = {k: global_model[k] +  local_models[idx][k] for k in global_model.keys()}
    global_model = {k: global_model[k] / args.num_groups for k in global_model.keys()}
    return global_model

def aggregation_adam(global_model, local_models, u, v, idxs_users, args, t):
    '''
    adam update on server side 
    '''
    lr_g = args.global_lr * 1/(t+1)**0.5
    # local只能用sgd 不能用momentum，不然要diverge
    # for i in idxs_users:
    #     local_models[i] = {k:local_models[i][k] - global_model[k] for k in global_model.keys()}
        
    model_avg = average(local_models, idxs_users)
    model_delta = {k:model_avg[k] - global_model[k] for k in model_avg.keys()}
    
    u = {k:(args.beta_1 * u[k] + (1-args.beta_1) * model_delta[k])for k in u.keys()}
    v = {k:(args.beta_2 * v[k] + (1-args.beta_2) * (u[k] * u[k])) for k in v.keys()}
    global_model = {k: global_model[k] + lr_g * u[k] / (torch.sqrt(v[k]) + args.kappa) for k in global_model.keys()}
    return global_model, u, v

def aggregation_privc(global_model, local_models, idxs_users, args, t):
    if args.privc == 'dp_randk' or args.privc == 'dp_topk' or args.privc == 'randk' or args.privc == 'topk':
        # error compensate, clipping, privately compress and average
        # lr_g = args.global_lr * 1/(t+1)**0.5
        pris_local_updates =[]
        norm_para=torch.zeros(args.num_users).to(args.device)
        for i in idxs_users:
            if (t+1) == args.tau:
                local_models[i] = {k:local_models[i][k] - global_model[k] for k in global_model.keys()}
                local_models[i], norm_para[i] = model_clip(local_models[i], args.clip)
            else:
                local_models[i] = {k:local_models[i][k] - global_model[k] + errors[i][k] for k in global_model.keys()}
                local_models[i], norm_para[i] = model_clip(local_models[i], args.clip)
                            
            pris_update = private_com(local_models[i], args)
            pris_local_updates.append(pris_update)
        print('weight norm', sum(norm_para)/len(idxs_users))
        
        global_update = {k: global_model[k] *0.0 for k in global_model.keys()}
        for i in len(idxs_users):
            global_update = {k: global_update[k] +  pris_local_updates[i][k] for k in global_update.keys()}
        
        global_update = {k: global_update[k] +  pris_local_updates[i][k] for k in global_update.keys()}
        global_model = {k: global_model[k] + global_update[k] for k in global_model.keys()}
    else:
        exit('Error: unrecognized private compressor for pefed.') 
        # global_model = aggregation_avg(local_models, idxs_users)
    return global_model