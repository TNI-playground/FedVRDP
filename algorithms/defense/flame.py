'''
Implementation of FLAME: Taming Backdoors in Federated Learning is from https://www.usenix.org/system/files/sec22-nguyen.pdf
Code from: https://github.com/zhmzm/FLAME

Learn how to deal with the parameters in more complicated models. 
'''


import numpy as np
import torch
import copy
import hdbscan

def no_defence_balance(params, global_parameters):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters

def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def flame(local_models, local_updates, global_model, args):
    # num_user = len(local_models)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_models:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    
    for i in range(args.num_selected_users):
        cos_i = []
        for j in range(args.num_selected_users):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            # cos_i.append(round(cos_ij.item(),4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    # num_clients = max(int(args.frac * args.num_users), 1) == num_users
    # num_malicious_clients = int(args.malicious * num_clients) # == args.num_attackers
    # num_benign_clients = args.num_selected_users - args.num_attackers
    clusterer = hdbscan.HDBSCAN(min_cluster_size=args.num_selected_users//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    # print(clusterer.labels_)
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_models)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(local_updates[i]),p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
                # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
                norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(local_updates[i]),p=2).item())  # no consider BN
    # print(benign_client)
   
    # for i in range(len(benign_client)):
    #     if benign_client[i] < args.num_attackers:
    #         args.wrong_mal+=1
    #     else:
    #         #  minus per benign in cluster
    #         args.right_ben += 1
    # args.turn+=1
    # if args.num_attackers > 0:
        # print('proportion of malicious are selected:',args.wrong_mal/(args.num_attackers*args.turn))
    # print('proportion of benign are selected:',args.right_ben/(num_benign_clients*args.turn))
    
    clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value/norm_list[i]
        if gama < 1:
            for key in local_updates[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                local_updates[benign_client[i]][key] *= gama
    global_model = no_defence_balance([local_updates[i] for i in benign_client], global_model)
    #add noise
    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
                    continue
        temp = copy.deepcopy(var)
        temp = temp.normal_(mean=0,std=args.noise*clip_value)
        # print('sigma:'+str(args.noise*clip_value))
        var += temp
    return global_model
    