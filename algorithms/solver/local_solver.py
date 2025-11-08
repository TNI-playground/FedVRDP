import torch
from torch import nn
import copy

from ..privacy.dp_compress import private_com, cpsgd
from ..attack.dba import dba_poison
from ..attack.edge import edge_poison


class LocalUpdate(object):
    def __init__(self, args):
        self.args = args
        if args.data_type == 'image':
            self.loss_func = nn.CrossEntropyLoss()
        elif args.data_type == 'text':
            self.loss_func = nn.CrossEntropyLoss()

    def sgd(self, net, samples, labels):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr, momentum=self.args.local_momentum)
        net.zero_grad()
        log_probs = net(samples)
        loss = self.loss_func(log_probs, labels)
        loss.backward()
        # update
        optimizer.step()
        w_new = copy.deepcopy(net.state_dict())
        return w_new, loss.item()
    
    def sgd_with_gradient_perturbation(self, net, samples, labels):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr, momentum=self.args.local_momentum)
        net.zero_grad()
        log_probs = net(samples)
        loss = self.loss_func(log_probs, labels)
        loss.backward()
        # grad_norm = nn.utils.clip_grad_norm_(net.parameters(), self.args.clip)
        # # print('gradient norm = {}'.format(grad_norm))
        # # if private compression on gradient like fedsap and dpfed
        # if 'grad' in self.args.privc:
        #     net = private_com(net, self.args)
        # elif self.args.method == 'cpsgd':
        #     _ = nn.utils.clip_grad_norm_(net.parameters(), self.args.clip)
        #     net = cpsgd(net, self.args)
        # else:
        #     pass;
        
        # update
        optimizer.step()
        w_new = copy.deepcopy(net.state_dict())
        return w_new, loss.item()
    
    def local_sgd(self, net, ldr_train, topk_model=None):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr)
        epoch_loss = []
        net.train()
        for _ in range(self.args.tau):
            for _, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
        w_new = copy.deepcopy(net.state_dict())
        return w_new, sum(epoch_loss) / len(epoch_loss)

    def local_sgd_mome(self, net, ldr_train, topk_model=None):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr, momentum=self.args.local_momentum)
        epoch_loss = []
        net.train()
        for _ in range(self.args.tau):
            for _, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
        w_new = copy.deepcopy(net.state_dict())
        return w_new, sum(epoch_loss) / len(epoch_loss)
    
    def local_sgd_adam(self, net, ldr_train, topk_model=None):
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.local_lr, betas=(0.5, 0.5))
        epoch_loss = []
        net.train()
        for _ in range(self.args.tau):
            for _, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
        w_new = copy.deepcopy(net.state_dict())
        return w_new, sum(epoch_loss) / len(epoch_loss)

    def local_sgd_with_dptopk(self, net, ldr_train, topk_model=None):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr, momentum=self.args.local_momentum)
        epoch_loss = []
        net.train()
        for _ in range(self.args.tau):
            for _, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                grads=[]
                for param in net.parameters():
                    grads.append(param.grad.view(-1))
                grads = torch.cat(grads)
                if self.args.privc == 'dptopk_new':
                    if topk_model:
                        for name, p in net.named_parameters():
                            topk_model[name][topk_model[name] != 0]=1
                            p.grad.detach().mul_(topk_model[name])
                optimizer.step()
                epoch_loss.append(loss.item())
        w_new = copy.deepcopy(net.state_dict())
        return w_new, sum(epoch_loss) / len(epoch_loss)

    def local_sgd_with_gradient_perturbation(self, net, ldr_train, topk_model=None):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr, momentum=self.args.local_momentum)
        epoch_loss = []
        net.train()
        # ldr_train = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        for _ in range(self.args.tau):
            for _, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # print(s)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                if 'grad' in self.args.privc:
                    _ = nn.utils.clip_grad_norm_(net.parameters(), self.args.clip)
                    net = private_com(net, self.args)
                # elif self.args.method == 'cpsgd':
                elif self.args.method == 'cpsgd':
                    _ = nn.utils.clip_grad_norm_(net.parameters(), self.args.clip)
                    net = cpsgd(net, self.args)
                else:
                    pass;        
                optimizer.step()
                epoch_loss.append(loss.item())
        w_new = copy.deepcopy(net.state_dict())
        return w_new, sum(epoch_loss) / len(epoch_loss)
    
    def local_sgd_with_dba(self, net, ldr_train, global_round=None, attack_cnt=None):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr, momentum=self.args.local_momentum)
        poison_optimizer = torch.optim.SGD(net.parameters(), lr=self.args.poison_local_lr, momentum=self.args.local_momentum)
        epoch_loss = []
        net.train()
        trigger_cnt = attack_cnt % self.args.trigger_num

        if (global_round in [self.args.poison_rounds]) or self.args.poison_rounds == -1:
            for _ in range(self.args.poison_tau):
                for _, (images, labels) in enumerate(ldr_train):
                    images, labels = dba_poison(images, labels, self.args, trigger_cnt)
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    poison_optimizer.step()
                    epoch_loss.append(loss.item())
        else:
            for _ in range(self.args.tau):
                for _, (images, labels) in enumerate(ldr_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
        w_new = copy.deepcopy(net.state_dict())
        return w_new, sum(epoch_loss) / len(epoch_loss)

    def local_sgd_with_edge(self, net, ldr_train, global_round=None, attack_cnt=None):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr, momentum=self.args.local_momentum)
        epoch_loss = []
        net.train()

        if (global_round in [self.args.poison_rounds]) or self.args.poison_rounds == -1:
            for _ in range(self.args.tau):
                for _, (images, labels) in enumerate(ldr_train):
                    images, labels = edge_poison(images, labels, self.args)
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
        else:
            for _ in range(self.args.tau):
                for _, (images, labels) in enumerate(ldr_train):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
        w_new = copy.deepcopy(net.state_dict())
        return w_new, sum(epoch_loss) / len(epoch_loss)