import copy
import numpy as np
import time, math
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data_pre_process import load_partition, DatasetSplit
from utils.model_utils import model_setup
from utils.log_utils import set_log_path
from test import test_img

from ..solver.local_solver import LocalUpdate
from ..solver.global_aggregator import average

from ..defense.byzantine_robust_aggregation import multi_krum, bulyan, tr_mean, median, coordinate_median
from ..defense.flame import flame
from ..defense.sparsefed import sparsefed

from ..defense.dpfed import dpfed
from ..defense.epfed import epfed
from ..defense.epfed_plus import epfed_plus
from ..defense.epfed_HBSCAN import epfed_HBSCAN
from ..defense.epfed_ablation_CR import epfed_ablation_CR
from ..defense.epfed_residualsparse import epfed_residualsparse
from ..defense.epfed_TemporalAttention import epfed_temporalattention


from ..attack.fang import fang_trmean_median_gray, fang_trmean_median_white, fang_krum_bulyan_gray
from ..attack.agr import agrAgnosticMinMax, agrAgnosticMinSum, agrTailoredTrmean, agrTailoredMedian, agrTailoredKrumBulyan

def attack_defense_fedavg(args):
    ################################### hyperparameter setup ########################################
    print("{:<50}".format("-" * 15 + " data setup " + "-" * 50)[0:60])
    # args, dataset_train, dataset_test, dataset_val, dataset_public, dict_users = load_partition(args)
    args, dataset_train, dataset_test, dataset_val, _, dict_users = load_partition(args)
    print('length of dataset:{}'.format(len(dataset_train) + len(dataset_test) + len(dataset_val)))
    print('num. of training data:{}'.format(len(dataset_train)))
    print('num. of testing data:{}'.format(len(dataset_test)))
    print('num. of validation data:{}'.format(len(dataset_val)))
    # print('num. of public data:{}'.format(len(dataset_public)))
    print('num. of users:{}'.format(len(dict_users)))
    sample_per_users = int(sum([ len(dict_users[i]) for i in range(len(dict_users))])/len(dict_users))
    print('average num. of samples per user:{}'.format(sample_per_users))
    
    print("{:<50}".format("-" * 15 + " log path " + "-" * 50)[0:60])
    log_path = set_log_path(args)
    writer = SummaryWriter(log_path)
    print(log_path)
    
    print("{:<50}".format("-" * 15 + " model setup " + "-" * 50)[0:60])
    args, net_glob, global_model, args.dim = model_setup(args)

    print('model dim:', args.dim)

    ###################################### model initialization ###########################
    t1 = time.time()
    train_loss, test_acc = [], []
    print("{:<50}".format("-" * 15 + " training... " + "-" * 50)[0:60])
    # initialize data loader for training and/or public dataset
    data_loader_list = []
    for i in range(args.num_users):
        dataset = DatasetSplit(dataset_train, dict_users[i])
        ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        data_loader_list.append(ldr_train)

    net_glob.train()

    best_test_accuracy = 0

    attack_flag = False
    defend_flag = False
    if hasattr(args, 'attack'):
        if args.attack != 'None':
            attack_flag = True
        else:
            args.attack = None
            args.num_attackers = 0
    else:
        args.attack = None
        args.num_attackers = 0
    
    if hasattr(args, 'defend'):
        if args.defend != 'None':
            defend_flag = True
        else:
            args.defend = None
    else:
        args.defend = None

    # sampling attackers' id
    if args.attack:
        attacked_idxs = list(np.random.choice(range(args.num_users), int(args.num_attackers/args.num_selected_users*args.num_users), replace=False))
    overall_attack_ratio = []

    for t in range(args.round):
        if args.attack:
            gt_attack_cnt = 0

        ## learning rate decaying
        if args.dataset == 'shakespeare':
            if (t+1) % 50 == 0:
                args.local_lr = args.local_lr * args.decay_weight
        else:
            args.local_lr = args.local_lr * args.decay_weight
        ## Server decaying
        if (t+1) % 10 == 0:
            if args.defend == 'epfed_plus':
                if args.use_momentum:
                    args.momentum_eta_g = args.momentum_eta_g * args.decay_weight
        ############################################################# FedAvg ##########################################
        ## user selection
        selected_idxs = list(np.random.choice(range(args.num_users), args.num_selected_users, replace=False))
        ## copy global model
        # net_glob.load_state_dict(global_model)
        ## local training
        # local_solver = LocalUpdate(args=args)
        local_models, local_losses, local_updates, malicious_updates, delta_norms= [], [], [], [], []
        
        for i in selected_idxs:
            ################## <<< Attack Point 1: train with poisoned data
            net_glob.load_state_dict(global_model)
            local_solver = LocalUpdate(args=args)

            if attack_flag and i in attacked_idxs:
                gt_attack_cnt += 1
                if args.attack == 'dba':
                    local_model, local_loss = local_solver.local_sgd_with_dba(
                        net=copy.deepcopy(net_glob).to(args.device),
                        ldr_train=data_loader_list[i],
                        global_round=t)#,
                        # attack_cnt=attack_cnt)
                elif args.attack == 'edge':
                    local_model, local_loss = local_solver.local_sgd_with_edge(
                        net=copy.deepcopy(net_glob).to(args.device),
                        ldr_train=data_loader_list[i])
                elif args.defend == 'epfed' or args.defend == 'epfed_plus':
                    local_model, local_loss = local_solver.local_sgd_mome(
                        net=copy.deepcopy(net_glob).to(args.device),
                        ldr_train=data_loader_list[i])
                else:
                    local_model, local_loss = local_solver.local_sgd(
                        net=copy.deepcopy(net_glob).to(args.device),
                        ldr_train=data_loader_list[i])
            else:
                if args.defend == 'epfed' or args.defend == 'epfed_plus':
                    local_model, local_loss = local_solver.local_sgd_mome(
                        net=copy.deepcopy(net_glob).to(args.device),
                        ldr_train=data_loader_list[i])
                else:
                    local_model, local_loss = local_solver.local_sgd(
                        net=copy.deepcopy(net_glob).to(args.device),
                        ldr_train=data_loader_list[i])
            
            local_losses.append(local_loss)
            # compute model update
            model_update = {k: local_model[k] - global_model[k] for k in global_model.keys()}
            # compute model update norm
            delta_norm = torch.norm(torch.cat([torch.flatten(model_update[k]) for k in model_update.keys()]))
            delta_norms.append(delta_norm)
            # clipping local model 
            if defend_flag:
                if args.defend in ['sparsefed', 'dpfed', 'tr_mean', 'coordinate_median', 'krum', 'bulyan', None]:
                    threshold = delta_norm / args.clip
                    if threshold > 1.0:
                        for k in model_update.keys():
                            model_update[k] = model_update[k] / threshold
            # collecting local models
            # 32 bits * args.dim, {(index, param)}: k*32+log2(d); 32->4; 
            if attack_flag and i in attacked_idxs:
                malicious_updates.append(model_update)
            else:
                local_updates.append(model_update)

            #
            if args.defend == 'flame':
                local_models.append(local_model)

        # add malicious update to the start of local updates
        local_updates = malicious_updates + local_updates
        # gt attack ratio
        if args.num_attackers > 0:
            gt_attack_ratio = gt_attack_cnt / args.num_selected_users
            print('current iteration attack ratio: '+str(gt_attack_ratio))
            overall_attack_ratio.append(gt_attack_ratio)

        # metrics
        # train_loss.append(sum(local_losses) / args.num_selected_users)
        # median_model_norm.append(torch.median(torch.stack(delta_norms)).cpu())
        norm = torch.median(torch.stack(delta_norms)).cpu()
        train_loss = sum(local_losses) / args.num_selected_users
        writer.add_scalar('norm', norm, t)
        writer.add_scalar('train_loss', train_loss, t)

        ################## <<< Attack Point 2: local model poisoning attacks
        if args.attack == 'fang_trmean_median_gray':
            local_updates = fang_trmean_median_gray(local_updates, args) # specify gray box or white box attack here
            # local_updates = fang_trmean_white(local_updates, args)
        elif args.attack == 'fang_krum_bulyan_gray':
            local_updates = fang_krum_bulyan_gray(local_updates, args)
        elif args.attack == 'agrAgnosticMinMax':
            local_updates = agrAgnosticMinMax(local_updates, args)
        elif args.attack == 'agrAgnosticMinSum':
            local_updates = agrAgnosticMinSum(local_updates, args)
        elif args.attack == 'agrTailoredTrmean':
            local_updates = agrTailoredTrmean(local_updates, args)
        elif args.attack == 'agrTailoredMedian':
            local_updates = agrTailoredMedian(local_updates, args)
        elif args.attack == 'agrTailoredKrumBulyan':
            local_updates = agrTailoredKrumBulyan(local_updates, args)
        
        ## robust/non-robust global aggregation
        if args.attack:
            print('attack:' + args.attack)
        else:
            print('attack: None')

        if args.defend:
            print('defend:' + args.defend)
        else:
            print('defend: None')

        if args.defend == 'multi_krum':
            aggregate_model, _ = multi_krum(local_updates, multi_k=True)
            global_model = average(global_model, [aggregate_model])
        elif args.defend == 'krum':
            aggregate_model, _ = multi_krum(local_updates, multi_k=False)
            global_model = average(global_model, [aggregate_model])
        elif args.defend == 'bulyan':
            aggregate_model, _ = bulyan(local_updates)
            global_model = average(global_model, [aggregate_model])
        elif args.defend == 'tr_mean':
            aggregate_model = tr_mean(local_updates)
            global_model = average(global_model, [aggregate_model])
        elif args.defend == 'coordinate_median':
            # aggregate_model = median(local_updates)
            aggregate_model = coordinate_median(local_updates)
            global_model = average(global_model, [aggregate_model])
        elif args.defend == 'flame':
            global_model = flame(local_models, local_updates, global_model, args)
        elif args.defend == 'sparsefed':
            if t > 0:
                global_model, momentum, error = sparsefed(local_updates, global_model, args, momentum, error)
            else:
                global_model, momentum, error = sparsefed(local_updates, global_model, args)
        elif args.defend == 'dpfed':
            global_model = dpfed(local_updates, global_model, args)
        elif args.defend == 'epfed':
            if t > 0:
                global_model, momentum, global_mask = epfed(local_updates, global_model, args, momentum, global_mask)
            else:
                global_model, momentum, global_mask = epfed(local_updates, global_model, args)
        elif args.defend == 'epfed_plus':
            if t > 0:
                global_model, momentum, momentum_tilde, global_mask = epfed_plus(local_updates, global_model, args, momentum, momentum_tilde, global_mask)
            else:
                global_model, momentum, momentum_tilde, global_mask = epfed_plus(local_updates, global_model, args)
        elif args.defend == 'epfed_residualsparse':
            if t > 0:
                global_model, momentum, global_mask = epfed_residualsparse(local_updates, global_model, args, momentum, global_mask)
            else:
                global_model, momentum, global_mask = epfed_residualsparse(local_updates, global_model, args)
        elif args.defend == 'epfed_temporalattention':
            if t > 0:
                global_model, momentum, global_mask = epfed_temporalattention(local_updates, global_model, args, momentum, global_mask)
            else:
                global_model, momentum, global_mask = epfed_temporalattention(local_updates, global_model, args)
        elif args.defend == 'epfed_ablation_CR':
            if t > 0:
                global_model, momentum, global_mask = epfed_ablation_CR(local_updates, global_model, args, momentum, global_mask)
            else:
                global_model, momentum, global_mask = epfed_ablation_CR(local_updates, global_model, args)
        else:
            global_model = average(global_model, local_updates) # just fedavg

        ## test global model on server side
        net_glob.load_state_dict(global_model)
        with torch.no_grad():
            test_acc, test_loss = test_img(net_glob, dataset_test, args)
        
        # metrics
        writer.add_scalar('test_acc', test_acc, t)
        print('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
              format(t, train_loss, norm, test_acc))
        
        if best_test_accuracy < test_acc:
            best_test_accuracy = test_acc
            # if args.defend == 'epfed_ablation_CR':
            #     np.savetxt('best_mask.txt', global_mask[0].cpu().numpy(), fmt='%d')
            #     print('save to best_mask.txt..')

        # stop
        if math.isnan(train_loss) or train_loss > 1e8 or t == args.round - 1:
            t2 = time.time()
            hours, rem = divmod(t2-t1, 3600)
            minutes, seconds = divmod(rem, 60)
            print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
            print("best test accuracy ", best_test_accuracy)
            if len(overall_attack_ratio) > 0:
                print("overall poisoned ratio ", str(np.average(overall_attack_ratio)))
                return best_test_accuracy, np.average(overall_attack_ratio)
            else:
                return best_test_accuracy, 0
