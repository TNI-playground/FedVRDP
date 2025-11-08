import torch
from torch import nn
from torch.utils.data import DataLoader
import copy
from algorithms.attack.dba import dba_poison
from algorithms.attack.edge import edge_poison

def test_img(net_g, datatest, args):
    net_g = copy.deepcopy(net_g).to(args.device)
    loss_func = nn.CrossEntropyLoss()
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.test_batch_size)

    # attack init
    attack_flag = False
    if hasattr(args, 'attack'):
        if args.attack != 'None':
            attack_flag = True
    if attack_flag:
        attack_ratio = args.num_attackers / args.num_selected_users
        poison_data_iteration = int(attack_ratio * len(data_loader))

    for index, (data, target) in enumerate(data_loader):
        ################## <<< Attack Point 3: trigger for all data during testing
        if attack_flag:
            if index < poison_data_iteration:
                if args.attack == 'dba':
                    trigger_cnt = index % args.trigger_num
                    data, target = dba_poison(data, target, args, trigger_cnt, evaluation=True)
                elif args.attack == 'edge':
                    data, target = edge_poison(data, target, args, evaluation=True)
                
        data, target = data.to(args.device), target.to(args.device)
        # print(data[0], target[0])
        
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += loss_func(log_probs, target).item()
        # get the index of the max log-probability
        # print(log_probs[0])
        # print(torch.max(log_probs[0], -2))
        # exit()
        if args.dataset == 'shakespeare':
            _, predicted = torch.max(log_probs, -2)
            # correct += predicted[:,-1].eq(target[:,-1]).sum()
            correct += predicted.eq(target).sum()/target.shape[1]
            # print(predicted[:,-1].eq(target[:,-1]))
        else:
            _, predicted = torch.max(log_probs, -1)
            correct += predicted.eq(target).sum()

    test_loss /= len(datatest)
    # print(correct, len(datatest))
    accuracy = 100.00 * correct.item() / len(datatest)
    # exit()
    return accuracy, test_loss