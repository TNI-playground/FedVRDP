from model.cnn import CNNFmnist, CNNSvhn, CNNCifar
from model.mlp import MLP
from model.recurrent import CharLSTM, RNN_FedShakespeare
from model.vgg import vgg19_bn
from model.resnet import ResNet9FashionMNIST, ResNet18, ReducedResNet18, \
                         ResNet34, ResNet50, ResNet101, ResNet152, \
                         CIFARResNet20, SVHNResNet20
import torch
import copy

################################### model setup ########################################
def model_setup(args):
    if args.model == 'mlp':
        len_in = 1
        for x in args.img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'cnncifar':
        net_glob = CNNCifar().to(args.device)
    elif args.model == 'cnnfmnist':
        net_glob = CNNFmnist(args=args).to(args.device)
    elif args.model == 'cnnsvhn':
        net_glob = CNNSvhn(args=args).to(args.device)
    # elif args.model == 'cnn' and args.dataset == 'mnist':
    #     net_glob = CNNMnist(args=args).to(args.device)
    #     # args.epochs = args.local_ep * 40
    # elif args.model == 'cnn' and args.dataset == 'emnist':
    #     net_glob = CNNMnist(args=args).to(args.device)
    #     # args.epochs = args.local_ep * 40
    elif args.model == 'resnet9fmnist':
        net_glob = ResNet9FashionMNIST().to(args.device)
    elif args.model == 'resnet18':
        net_glob = ResNet18().to(args.device)
    elif args.model == 'reducedresnet18':
        net_glob = ReducedResNet18().to(args.device)
    elif args.model == 'resnet20svhn':
        net_glob = SVHNResNet20().to(args.device)
    elif args.model == 'resnet20cifar':
        net_glob = CIFARResNet20().to(args.device)
    # elif args.model == 'resnet34':
    #     net_glob = ResNet34().to(args.device)
    # elif args.model == 'resnet50':
    #     net_glob = ResNet50().to(args.device)
    # elif args.model == 'resnet101':
    #     net_glob = ResNet101().to(args.device)
    # elif args.model == 'resnet152':
    #     net_glob = ResNet152().to(args.device)
    elif args.model == 'VGG' and args.dataset == 'cifar':
        net_glob = vgg19_bn().to(args.device)
    # e;if args.model == 'vgg11':
    #     net_glob = vgg11().to(args.device)
    # elif args.model == 'vgg13':
    #     net_glob = vgg13().to(args.device)
    # elif args.model == 'vgg16':
    #     net_glob = vgg16().to(args.device)
    # elif args.model == 'vgg19':
    #     net_glob = vgg19().to(args.device)
    elif args.model == 'rnnshakespeare':
        net_glob = RNN_FedShakespeare().to(args.device)
    elif args.model == 'rlr_mnist' and args.dataset == 'fmnist':
        net_glob = CNNFmnist(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    global_model = copy.deepcopy(net_glob.state_dict())
    return args, net_glob, global_model, model_dim(global_model)

def model_dim(model):
    '''
    compute model dimension
    '''
    flat = [torch.flatten(model[k]) for k in model.keys()]
    s = 0
    for p in flat: 
        s += p.shape[0]
    return s


def model_clip(model, clip):
    '''
    clip model update
    '''
    model_norm=[]
    for k in model.keys():
        if 'num_batches_tracked' in k or 'running_mean' in k or 'running_var' in k:
            continue
        model_norm.append(torch.norm(model[k]))
        
    total_norm = torch.norm(torch.stack(model_norm))
    clip_coef = clip / (total_norm + 1e-8)
    if clip_coef < 1:
        for k in model.keys():
            if 'num_batches_tracked' in k or 'running_mean' in k or 'running_var' in k:
                continue
            model[k] = model[k] * clip_coef
    return model, total_norm

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


def get_trainable_values(net,mydevice=None):
    ' return trainable parameter values as a vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable) 
    N=0
    for params in paramlist:
        N+=params.numel()
    if mydevice:
        X=torch.empty(N,dtype=torch.float).to(mydevice)
    else:
        X=torch.empty(N,dtype=torch.float)
    X.fill_(0.0)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            X[offset:offset+numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset+=numel

    return X

def put_trainable_values(net,X):
    ' replace trainable parameter values by the given vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset+=numel
