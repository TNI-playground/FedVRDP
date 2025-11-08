import numpy as np
import os
import dill
import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class custom_subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The subset Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.targets = labels
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

# split for federated settings
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


################################### data setup ########################################
def load_partition(args):
    dict_users = []
    # read dataset
    if args.dataset == 'mnist':
        path = './data/dataset/mnist'
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.MNIST(path, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(path, train=False, download=True, transform=trans_mnist)
        args.num_classes = 10
        # split training dataset (iid or non-iid)
        # remove dict_users.pik at the first time. 

        pik_name = args.config_name.split('/')[-1].split('.')[0]
        pik_path = os.path.join(path, pik_name+'_dict_users.pik')
        if os.path.isfile(pik_path):
            with open(pik_path, 'rb') as f: 
                dict_users = dill.load(f) 
        if len(dict_users) < 1:
            if args.iid:
                dict_users = iid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
            else:
                dict_users = noniid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
            
    elif args.dataset == 'svhn':
        path = './data/dataset/svhn'
        trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.43090966, 0.4302428, 0.44634357), (0.19759192, 0.20029082, 0.19811132))])
        dataset_train = datasets.SVHN(path, split='train', download=True, transform=trans_svhn)
        dataset_extra = datasets.SVHN(path, split='extra', download=True, transform=trans_svhn)
        dataset_train = torch.utils.data.ConcatDataset([dataset_train, dataset_extra])
        dataset_test = datasets.SVHN(path, split='test', download=True, transform=trans_svhn)
        args.num_classes = 10

        pik_name = args.config_name.split('/')[-1].split('.')[0]
        pik_path = os.path.join(path, pik_name+'_dict_users.pik')
        if os.path.isfile(pik_path):
            with open(pik_path, 'rb') as f: 
                dict_users = dill.load(f) 
        if len(dict_users) < 1:
            if args.iid:
                dict_users = iid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
            else:
                exit('Error: only consider IID setting in SVHN')

    elif args.dataset == 'emnist':
        path = './data/dataset/emnist'
        trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))])
        dataset_train = datasets.EMNIST(path, split='balanced', train=True, download=True, transform=trans_emnist)
        dataset_test = datasets.EMNIST(path, split='balanced', train=False, download=True, transform=trans_emnist)
        args.num_classes = 10

        pik_name = args.config_name.split('/')[-1].split('.')[0]
        pik_path = os.path.join(path, pik_name+'_dict_users.pik')
        if os.path.isfile(pik_path):
            with open(pik_path, 'rb') as f: 
                dict_users = dill.load(f) 
        if len(dict_users) < 1:
            if args.iid:
                dict_users = iid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
            else:
                exit('Error: only consider IID setting in emnist')

    elif args.dataset == 'fmnist':
        path = './data/dataset/fmnist'
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST(path, train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST(path, train=False, download=True, transform=trans_fmnist)
        args.num_classes = 10

        # pik_name = args.config_name.split('/')[-1].split('.')[0]
        # pik_path = os.path.join(path, pik_name+'_dict_users.pik')
        pik_path = os.path.join(path,'fmnist_dict_users.pik')
        if os.path.isfile(pik_path):
            with open(pik_path, 'rb') as f: 
                dict_users = dill.load(f) 
        if len(dict_users) < 1:
            if args.iid:
                dict_users = iid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
            else:
                dict_users = noniid(dataset_train, args.num_users, class_num=args.noniid_clsnum)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)

    elif args.dataset == 'cifar':
        path = './data/dataset/cifar'
        dataset_train = datasets.CIFAR10(path, train=True, download=True, 
                                        transform=transforms.Compose([transforms.RandomHorizontalFlip(), 
                                                                        transforms.RandomCrop(32, 4),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225])]))
        dataset_test = datasets.CIFAR10(path, train=False, download=True, 
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                            std=[0.229, 0.224, 0.225])]))
        args.num_classes = 10

        pik_name = args.config_name.split('/')[-1].split('.')[0]
        pik_path = os.path.join(path, pik_name+'_dict_users.pik')
        if os.path.isfile(pik_path):
            with open(pik_path, 'rb') as f: 
                dict_users = dill.load(f) 
        if len(dict_users) < 1:
            if args.iid:
                dict_users = iid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
            else:
                exit('Error: only consider IID setting in CIFAR10')
    
    elif args.dataset == 'shakespeare':
        dataset_train = torch.load('./data/dataset/shakespeare/train.pt')
        dataset_test = torch.load('./data/dataset/shakespeare/test.pt')
        dict_users = torch.load('./data/dataset/shakespeare/dict_users.pt')
        # dict_users = dataset_train.get_client_dic()
        # print(len(dataset_train), len(dataset_test))
        args.num_users = len(dict_users)
        if args.iid:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    
    ## extract 10% data from test set for validation, and the rest for testing
    print("Creating validation dataset from testing dataset...")
    dataset_test, dataset_val = torch.utils.data.random_split(dataset_test, [len(dataset_test)-int(0.1 * len(dataset_test)), int(0.1 * len(dataset_test))])
    ## generate a public dataset for DP-topk purpose from validation set
    dataset_test, dataset_val = dataset_test.dataset, dataset_val.dataset
    # print("Creating public dataset...")
    # dataset_public = public_iid(dataset_val, args) # make sure public set has every class
    ## make sure experiments with different sizes of public dataset use the same testing data and training data

    # return args, dataset_train, dataset_test, dataset_val, dataset_public, dict_users
    return args, dataset_train, dataset_test, dataset_val, None, dict_users

###################### utils #################################################
## IID assign data samples for num_users (mnist, svhn, fmnist, emnist, cifar)
def iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    print("Assigning training data samples (iid)")
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

## IID assign data samples for num_users (mnist, emnist, cifar); each user only has n(default:two) classes of data
def noniid(dataset, num_users, class_num=2):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: each user only has two classes of data
    """
    print("Assigning training data samples (non-iid)")
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, class_num, replace=False))
        if num_users <= num_shards:
            idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

## generate a iid public dataset from dataset. 
def public_iid(dataset, args):
    """
    Sample I.I.D. public data from fashion MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    if args.dataset == 'fmnist':
        labels = dataset.train_labels.numpy()
    elif args.dataset == 'cifar':
        labels = np.array(dataset.targets)
    else:
        labels = dataset.labels
    pub_set_idx = set()
    if args.pub_set > 0:
        for i in list(set(labels)):
            pub_set_idx.update(
                set(
                np.random.choice(np.where(labels==i)[0],
                                          int(args.pub_set/len(list(set(labels)))), 
                                 replace=False)
                )
                )
    # test_set_idx = set(np.arange(len(labels)))
    # test_set_idx= test_set_idx.difference(val_set_idx)
    return DatasetSplit(dataset, pub_set_idx)

def sample_dirichlet_train_data(dataset, args, no_participants, alpha=0.9):
    """
        Input: Number of participants and alpha (param for distribution)
        Output: A list of indices denoting data in CIFAR training set.
        Requires: cifar_classes, a preprocessed class-indice dictionary.
        Sample Method: take a uniformly sampled 10-dimension vector as parameters for
        dirichlet distribution to sample number of images in each class.
    """
    cifar_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if ind in args.poison_images or ind in args.poison_images_test:
            continue
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]
    class_size = len(cifar_classes[0])
    per_participant_list = {}
    no_classes = len(cifar_classes.keys())

    for n in range(no_classes):
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
            if user in per_participant_list:
                per_participant_list[user].extend(sampled_list)
            else:
                per_participant_list[user] = sampled_list
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

    return per_participant_list
