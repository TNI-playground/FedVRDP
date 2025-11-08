import copy
import numpy as np
import time, math
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data_pre_process import load_partition, DatasetSplit
from utils.model_utils import model_setup
from test import test_img

from utils.log_utils import set_log_path
from ..solver.local_solver import LocalUpdate
from ..solver.global_aggregator import average


def fedavg(args):
    ################################### hyperparameter setup ########################################
    
    print("{:<50}".format("-" * 15 + " data setup " + "-" * 50)[0:60])
    args, dataset_train, dataset_test, dataset_val, dataset_public, dict_users = load_partition(args)
    print('length of dataset:{}'.format(len(dataset_train) + len(dataset_test) + len(dataset_val)))
    print('num. of training data:{}'.format(len(dataset_train)))
    print('num. of testing data:{}'.format(len(dataset_test)))
    print('num. of validation data:{}'.format(len(dataset_val)))
    print('num. of public data:{}'.format(len(dataset_public)))
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
    train_loss, test_acc, median_model_norm = [], [], []
    print("{:<50}".format("-" * 15 + " training... " + "-" * 50)[0:60])
    # initialize data loader for training and/or public dataset
    data_loader_list = []
    for i in range(args.num_users):
        dataset = DatasetSplit(dataset_train, dict_users[i])
        ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        data_loader_list.append(ldr_train)

    net_glob.train()
    for t in range(args.round):
        ## learning rate decaying
        if args.dataset == 'shakespeare':
            if t+1 % 50 == 0:
                args.local_lr = args.local_lr * args.decay_weight
        else:
            args.local_lr = args.local_lr * args.decay_weight
        ############################################################# FedAvg ##########################################
        
        ## user selection
        selected_idxs = list(np.random.choice(range(args.num_users), max(int(args.frac * args.num_users), 1), replace=False))
        ## copy global model
        net_glob.load_state_dict(global_model)
        ## local training
        local_solver = LocalUpdate(args=args)
        local_losses, local_updates, delta_norms= [], [], []
        for i in selected_idxs:
            local_model, local_loss = local_solver.local_sgd(
                net=copy.deepcopy(net_glob).to(args.device),
                ldr_train=data_loader_list[i])
            local_losses.append(local_loss)
            # compute model update
            model_update = {k: local_model[k] - global_model[k] for k in global_model.keys()}
            # compute model update norm
            delta_norm = torch.norm(torch.cat([torch.flatten(model_update[k])for k in model_update.keys()]))
            delta_norms.append(delta_norm)
            # clipping local model 
            threshold = delta_norm / args.clip
            if threshold > 1.0:
                for k in model_update.keys():
                    model_update[k] = model_update[k] / threshold
            # collecting local models
            # 32 bits * args.dim, {(index, param)}: k*32+log2(d); 32->4; 
            local_updates.append(model_update)

        # metrics
        # train_loss.append(sum(local_losses) / args.num_selected_users)
        # median_model_norm.append(torch.median(torch.stack(delta_norms)).cpu())
        norm = torch.median(torch.stack(delta_norms)).cpu()
        train_loss = sum(local_losses) / args.num_selected_users
        writer.add_scalar('norm', norm, t)
        writer.add_scalar('train_loss', train_loss, t)

        ## global aggregation
        global_model = average(global_model, local_updates)

        ## test global model on server side
        net_glob.load_state_dict(global_model)
        net_glob.eval()
        test_acc, test_loss = test_img(net_glob, dataset_test, args)
        
        # metrics
        writer.add_scalar('test_acc', test_acc, t)
        # test_acc.append(test_acc_t)
        print('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
              format(t, train_loss, norm, test_acc))
        
        # stop
        if math.isnan(train_loss) or train_loss > 1e8 or t == args.round - 1:
            # np.savetxt(log_path + "_test_acc" + ".csv",
            #             test_acc,
            #             delimiter=",")
            # np.savetxt(log_path + "_train_loss" + ".csv",
            #             train_loss,
            #             delimiter=",")
            # np.savetxt(log_path + "_norm_" + ".csv", median_model_norm, delimiter=",")
            t2 = time.time()
            hours, rem = divmod(t2-t1, 3600)
            minutes, seconds = divmod(rem, 60)
            print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
            exit()

    
# if __name__ == '__main__':
    
#     fedavg()
