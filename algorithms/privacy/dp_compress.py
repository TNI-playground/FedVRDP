import torch
import copy, scipy
import random
from .rdp_accountant import *


def private_com(model, args):
    '''
    private compressor used on model update:
        topk on model update
        biased randk on model update
        unbiased randk on gradient
        gaussian mechanism (on gradient or not)
        dp-topk on model update
        dp-randk on model update
        unbiased dp-randk on gradient
        cpsgd
        non
    '''
    if args.privc == 'randk':
        model = randk(model, args)
    elif args.privc == 'randk_grad':
        model = randk_grad(model, args)
    elif args.privc == 'topk':
        model = topk(model, args)
    # elif args.privc == 'topk_grad':
    #     model = topk_grad(model, args)
    elif args.privc == 'gaussian_mechanism':
        model = gaussian_mechanism(model, args)
    elif args.privc == 'gaussian_mechanism_grad':
        model = gaussian_mechanism_grad(model, args)
    elif args.privc == 'dp_topk':
        model = dp_topk(model, args)
    elif args.privc == 'dp_randk':
        model = dp_randk(model, args)
    # elif args.privc == 'dp-topk-grad':
    #     model = dp_topk(model, args)
    elif args.privc == 'dp_randk_grad':
        model = dp_randk_grad(model, args)
    elif args.privc == 'cpsgd':
        model = cpsgd(model, args)
    elif args.privc == 'non':
        pass
    else:
        exit('Error: unrecognized private compressor.')
    return model


def gaussian_mechanism(model, args):
    '''
    gaussian mechanism on model update
    '''
    model = {k: model[k] + torch.normal(0, args.sigma, model[k].size()).to(args.device)
        for k in model.keys()}
    return model

def topk_local(model, indices, idx):
    '''
    topk on model update
    '''
    compressed = [torch.flatten(model[k]) for k in model.keys()]
    flat = torch.cat(compressed).view(1, -1)
    flat = flat.flatten()
    flat_com = torch.zeros_like(flat)
    flat_com[indices] = flat[indices]
    compressed_model = {k: flat_com[s:d].reshape(model[k].shape) for k, (s, d) in zip(model.keys(), idx)}
    return compressed_model

def dp_topk_local(model, args, indices, idx):
    '''
    topk on model update
    '''
    compressed = [torch.flatten(model[k]) for k in model.keys()]
    flat = torch.cat(compressed).view(1, -1)
    flat = flat.flatten() 
    # add noise
    flat = flat + torch.normal(0, args.sigma, flat.size()).to(args.device)
    flat_com = torch.zeros_like(flat)
    flat_com[indices] = flat[indices]
    compressed_model = {k: flat_com[s:d].reshape(model[k].shape) for k, (s, d) in zip(model.keys(), idx)}
    return compressed_model

def topk(model, args):
    '''
    topk on model update
    '''
    compressed = [torch.flatten(model[k]) for k in model.keys()]
    idx = []
    s = 0
    for p in compressed:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    k_dim = int(args.com_p * s)
    flat = torch.cat(compressed).view(1, -1)
    flat = flat.flatten()
    mask = torch.zeros_like(flat)
    flat_abs = abs(flat)
    _, indices = torch.topk(flat_abs, k_dim)
    # generate a mask, set topk as 1, otherwise 0
    mask[indices] = 1
    mask = {k: mask[s:d].reshape(model[k].shape) for k, (s, d) in zip(model.keys(), idx)}
    return mask

def sparsify(model, mask):
    for k in model.keys():
        model[k] = model[k] * mask[k]
    return model


def randk(model, args):
    '''
    biased randk on model update
    '''
    if args.com_p >= 1.0:
        print('compression ratio should less than 1.0')
        return
    else:
        compressed = [torch.flatten(model[k]) for k in model.keys()]
        idx = []
        s = 0
        for p in compressed:
            d = p.shape[0]
            idx.append((s, s + d))
            s += d
        k_dim = int(args.com_p * s)
        flat = torch.cat(compressed).view(1, -1)
        flat = flat.flatten()
        mask = torch.zeros_like(flat)
        perm = torch.randperm(s)
        idxx = perm[:int(k_dim)]
        mask[idxx] = 1
        mask = {
            k: mask[s:d].reshape(model[k].shape)
            for k, (s, d) in zip(model.keys(), idx)}
        return mask

def dp_randk_local(model, args, perm, k_dim, idx):
    '''
    biased randk on model update
    '''
    if args.com_p >= 1.0:
        return model
    else:
        compressed = [torch.flatten(model[k]) for k in model.keys()]
        flat = torch.cat(compressed).view(1, -1)
        flat = flat.flatten()
        # add noise 
        flat = flat + torch.normal(0, args.sigma, flat.size()).to(args.device)
        len_flat = len(flat)
        idxx = perm[:int((1 - k_dim / len_flat) * len_flat)]
        flat[idxx] = 0.0
        compressed_model = {k: flat[s:d].reshape(model[k].shape)for k, (s, d) in zip(model.keys(), idx)}
        return compressed_model


def dp_topk(model_update, args):
    '''
    biased dp-topk on model update
    '''
    model = copy.deepcopy(model_update)
    compressed = [torch.flatten(model[k]) for k in model.keys()]
    flat = torch.cat(compressed).view(1, -1)
    flat = flat.flatten()

    idx = []
    s = 0
    for p in compressed:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d
    k_dim = int(args.com_p * s)

    if args.topk_index == 'exponential':
        # delta_norm = torch.norm(flat)
        # threshold = delta_norm/args.clip_top
        # if threshold > 1.0:
        #     flat = flat / threshold
        flat_abs = abs(flat)
        sorted, indices = torch.sort(flat_abs, descending=False)
        print(indices)
        com = torch.exp(0.5 * (indices + 1) / (2 * s))
        print(com)
        # print(torch.sum(torch.exp(args.eps/k_dim * args.topk_ratio * indices/ (2 * s))/dom))
        probs = com / torch.sum(com)
        print(probs)
        # print(max(probs), min(probs))
        # exit()
        selected_idx = torch.nonzero(torch.bernoulli(probs).to(args.device))
        print(selected_idx)
        exit()

    if args.topk_index == 'gm':
        delta_norm = torch.norm(flat)
        threshold = delta_norm / args.clip_top
        if threshold > 1.0:
            flat = flat / threshold
        # clipping
        # 最后算选中的K个element的 l2 norm 为sensitivity，这样的话sensitivity< topk的l2 norm， 用initial的值代替
        # values, indices = torch.topk(abs(flat), k_dim+1)
        # print(indices)
        # print('l2 norm of topk={:3f}, median of topk={:3f}, average of topk={:3f}'.format(torch.norm(values), torch.median(values), torch.mean(values)))
        # flat = torch.clamp(flat, -args.clip_top, args.clip_top)

        flat_abs = abs(flat)
        values, indices = torch.topk(flat_abs, k_dim + 1)
        T = (values[-1] + values[-2]) / 2
        # print(indices)
        # print('l2 norm of topk={:3f}, median of topk={:3f}, average of topk={:3f}'.format(torch.norm(values), torch.median(values), torch.mean(values)))
        # exit('test')

        # true value
        compressed_model_update = [
            torch.flatten(model_update[k]) for k in model_update.keys()
        ]
        flat_model_update = torch.cat(compressed_model_update).view(1, -1)
        flat_model_update = flat_model_update.flatten()
        flat_com = torch.zeros_like(flat)
        # values, indices = torch.topk(abs(flat_model_update), k_dim+1)
        # print(indices)
        # print('l2 norm of topk={:3f}, median of topk={:3f}, average of topk={:3f}'.format(torch.norm(values), torch.median(values), torch.mean(values)))
        # exit('check')
        # print(torch.norm(torch.cat([torch.flatten(model_update[k]) for k in model_update.keys()])))

        selected_idx = []
        dim_left = k_dim
        selected_idx_ = []
        if args.sigma_1 > 0 and args.sigma_2 > 0:
            while len(selected_idx) < k_dim:
                a = torch.normal(0, args.sigma_2,
                                 flat_abs.size()).to(args.device)
                b = torch.normal(0, args.sigma_1,
                                 flat_abs.size()).to(args.device)
                satisfied_idx = (a + flat_abs > T + b).nonzero()
                satisfied_idx = satisfied_idx
                len_idx = len(satisfied_idx)
                if len_idx >= dim_left:
                    selected_idx_.append(
                        satisfied_idx[torch.randperm(len_idx)][:dim_left])
                else:
                    selected_idx_.append(satisfied_idx)
                    flat_abs[satisfied_idx] = -1e10
                    dim_left -= len_idx
                    # print(flat_abs.size())
                selected_idx = torch.cat(selected_idx_)
        else:
            selected_idx = (flat_abs > T).nonzero()
            # print(selected_idx_)

            # print(len(satisfied_idx), len(selected_idx), len(torch.unique(selected_idx)))
        # exit('true')
        selected_idx = selected_idx.flatten()
        flat_com[selected_idx] = flat_model_update[selected_idx]
        # print(len(flat_com.nonzero()))
        # exit('true')
        # print(flat_com.nonzero())

    com_model = {
        k: flat_com[s:d].reshape(model[k].shape)
        for k, (s, d) in zip(model.keys(), idx)
    }
    return com_model


def dp_randk(model, args):
    '''
    baised dp-randk on model update
    '''

    compressed = [torch.flatten(model[k]) for k in model.keys()]
    idx = []
    s = 0
    for p in compressed:
        d = p.shape[0]
        idx.append((s, s + d))
        s += d

    k_dim = int(args.com_p * s)
    flat = torch.cat(compressed).view(1, -1)
    flat = flat.flatten()
    len_flat = len(flat)

    perm = torch.randperm(len_flat)
    idxx = perm[:int((1 - k_dim / len_flat) * len_flat)]

    flat_com = flat + torch.normal(0, args.sigma, flat.size())
    flat_com[idxx] = 0.0
    compressed_noisy_model = {
        k: flat_com[s:d].reshape(model[k].shape)
        for k, (s, d) in zip(model.keys(), idx)
    }
    return compressed_noisy_model


def cpsgd(net, args):
    '''
    cpsgd: binomial mechanism with randomized quantization
    '''
    if args.quant_k > 0:
        for p in net.parameters():
            clip_thresh = torch.max(torch.abs(p.grad.detach()))
            # print('clip', clip_thresh)
            # clip_thresh = args.clip
            B = -clip_thresh + 2 * torch.arange(args.quant_k + 1).to(
                args.device) * clip_thresh / (args.quant_k - 1)
            B = B.to(p.grad.device)
            flat = torch.flatten(p.grad.detach())
            B = torch.zeros((len(B), len(flat))).to(p.grad.device) + B.reshape(
                (-1, 1))
            c = B == 0
            d = B * (B >= flat)
            d[d == 0.0] = float("Inf")
            d[B == 0.0] = 0.0
            r_1, _ = torch.min(d, 0)
            r_1[r_1 == float("Inf")] = B[0, -1]
            r_1[r_1 == float("NaN")] = B[0, -1]

            f = B * (B < flat)
            f[f == 0] = -float("Inf")
            f[B == 0.0] = 0.0
            r_2, _ = torch.max(f, 0)
            r_2[r_2 == -float("Inf")] = B[0, 0]
            r_2[r_2 == float("NaN")] = B[0, 0]

            h = (flat - r_2) / (r_1 - r_2)
            h[h != h] = 1.0
            h = h.reshape((-1, 1))
            hh = torch.cat((h, 1 - h), 1)
            # print(hh)
            # exit('Error: unrecognized method')

            x = torch.bernoulli(hh)
            flat_ = r_1 * x[:, 0] + r_2 * x[:, 1]
            # print(max(flat_))
            U = torch.bernoulli(
                torch.ones((len(flat), args.quant_m)).to(args.device) * 0.5)
            # print(len(torch.sum(U, 1)) == len(flat_))
            flat = flat_ + (2 * clip_thresh *
                            (torch.sum(U, 1) - args.quant_m * 0.5) /
                            (args.quant_k - 1))
            # print(max(flat))
            # print(sum(flat-flat_))
            p.grad.detach().add_(-p.grad.detach() +
                                 torch.reshape(flat, p.grad.shape))
    return net


def gaussian_mechanism_grad(net, args):
    '''
    gaussian mechanism on gradient 
    '''
    for p in net.parameters():
        p.grad.detach().add_(
            torch.normal(0, args.sigma, p.grad.size()).to(p.grad.device))
    return net


def randk_grad(net, args):
    '''
    unbiased randk on gradient
    '''
    torch.cuda.manual_seed(args.seed)
    sparsifier = copy.deepcopy(net.state_dict())
    # for k in sparsifier.keys():
    #     sparsifier[k] = torch.bernoulli(torch.zeros(sparsifier[k].size(), device=args.device) + args.com_p)/args.com_p
    # keys=list(sparsifier.keys())
    for p in net.parameters():
        sparsifier = torch.bernoulli(
            torch.zeros(p.grad.size(), device=p.grad.device) + args.com_p)
        # p.grad.detach().add_(torch.normal(0, args.sigma, p.grad.size()).to(p.grad.device))
        p.grad.detach().mul_(sparsifier)
    return net


def dp_randk_grad(net, args):
    '''
    unbiased randk on gradient
    '''
    secure_generator = torch.cuda.manual_seed(args.seed)
    sparsifier = copy.deepcopy(net.state_dict())
    for k in sparsifier.keys():
        sparsifier[k] = torch.bernoulli(
            torch.zeros(sparsifier[k].size(), device=args.device) + args.com_p,
            secure_generator) / args.com_p
    keys = list(sparsifier.keys())
    k = 0
    for p in net.parameters():
        p.grad.detach().add_(
            torch.normal(0, args.sigma, p.grad.size()).to(p.grad.device))
        p.grad.detach().mul_(sparsifier[keys[k]].to(p.grad.device))
        k += 1
    return net


def compute_eps(args, iter):
    if args.method == 'cpsgd':
        epsilon, _ = compute_eps_cpsgd(
            args.quant_k, args.quant_m, args.num_users * args.frac,
            0.5 * args.delta / (args.batch_size / args.num_data),
            args.batch_size / args.num_data, (iter + 1) * args.frac, args.dim)

    if args.privc == 'dprandk' or args.privc == 'gm' or args.privc == 'dpfed':
        epsilon = 0
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        rdp = compute_rdp(args.frac, 1, args.noise_multiplier, (iter + 1),
                          orders)
        epsilon = get_privacy_spent(orders, rdp, target_delta=args.delta)[0]
    elif args.privc == 'dptopk' or args.privc == 'dptopk_new':
        if 1:
            epsilon = 0
            orders = [1 + x / 10.0
                      for x in range(1, 100)] + list(range(12, 64))
            rdp = compute_rdp(args.frac, 1, args.noise_multiplier, (iter + 1),
                              orders)
            epsilon = get_privacy_spent(orders, rdp,
                                        target_delta=args.delta)[0]
        elif args.topk_ratio >= 1:
            # compute rdp2 first:
            orders = [1 + x / 10.0
                      for x in range(1, 100)] + list(range(12, 64))
            rdp = compute_rdp(args.frac, 1, args.noise_multiplier_1,
                              (iter + 1), orders)
            # print([scipy.special.comb(args.dim, j) for j in range(1, int(args.dim*args.com_p)+1)])
            for j in range(1, int(args.dim * args.com_p)):
                print(
                    math.log(scipy.special.comb(int(args.dim * args.com_p),
                                                j)))
                exit()
            rdp = rdp * (1 + args.topk_ratio) + compute_rdp(
                args.frac, 1, args.noise_multiplier_1 * 10,
                (iter + 1), orders) + math.log(
                    sum([
                        scipy.special.comb(args.dim, j)
                        for j in range(int(args.dim * args.com_p))
                    ])) / (orders - 1)
            epsilon = get_privacy_spent(orders, rdp,
                                        target_delta=args.delta)[0]
        else:
            # compute rdp1 first
            orders = [1 + x / 10.0
                      for x in range(1, 100)] + list(range(12, 64))
            rdp = compute_rdp(args.frac, 1, args.noise_multiplier, (iter + 1),
                              orders)
            rdp = rdp + rdp / args.topk_ratio + compute_rdp(
                args.frac, 1, args.noise_multiplier_1 * 10,
                (iter + 1), orders) + np.log(
                    sum([
                        scipy.special.comb(args.dim, j)
                        for j in range(int(args.dim * args.com_p))
                    ])) / (orders - 1)
            epsilon = get_privacy_spent(orders, rdp,
                                        target_delta=args.delta)[0]

        # # rdp2 =
        # rdp = rdp / args.topk_ratio
        # epsilon = get_privacy_spent(orders, rdp, target_delta=args.delta)[0]

        # eps2=0
        # rdp = compute_rdp(args.frac, 1, args.noise_multiplier_1, (iter+1), orders)
        # eps2 = get_privacy_spent(orders, rdp, target_delta=args.delta)[0]
        # epsilon = eps1 + eps2
    return epsilon
