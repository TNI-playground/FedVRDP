import torch

def lie_attack(all_updates, args, z=0.72575):
    # flatten attacker's model parameters only
    all_attack_updates_flatten=[]
    for update in all_updates[:args.num_attackers]:
        update = torch.cat([torch.flatten(update[k])for k in update.keys()])
        all_attack_updates_flatten = update[None, :] if not len(all_attack_updates_flatten) else torch.cat((all_attack_updates_flatten, update[None, :]), 0)

    avg = torch.mean(all_attack_updates_flatten, 0)
    std = torch.std(all_attack_updates_flatten, dim=0)
    mal_update = avg + z * std

    # use the same poisoned updates
    # mal_update = mal_update.unsqueeze(0).repeat(args.num_attackers, 1)
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
        all_updates[i] = {k: mal_update[i,:][s:d].reshape(all_updates[-1][k].shape) for k, (s, d) in zip(all_updates[-1].keys(), idx)}

    return all_updates