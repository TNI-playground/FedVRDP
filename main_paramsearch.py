import torch, random
from algorithms.engine.fedsgd import *
from algorithms.engine.fedavg import *
from algorithms.engine.byzantine_robust_fedavg import *
from algorithms.engine.byzantine_attack_fedavg_tune import *

from mmengine.config import Config
import optuna
from optuna.trial import TrialState
import logging
import sys

def merge_config(config, args):
    for arg in vars(args):
        setattr(config, arg, getattr(args, arg))
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
                        type=int,
                        default=7,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="seed")
    parser.add_argument('--repeat', type=int, default=5, help='repeat index')
    parser.add_argument('--freeze_datasplit', type=int, default=1, help='freeze to save dict_users.pik or not')
    # Modify following configure to test with different settings 
    parser.add_argument('--config_name', type=str,
                        default='attack/fmnist/fedsgd_image_fmnist_epfed.yaml',
                        help='attack, defend, privacy, compression, and reinforcement learning method configuration')
    parser.add_argument('--method',
                type=str,
                default='backdoor_attack_fedavg_tune',
                help='method name')

    args = parser.parse_args()
    config_path = os.path.join('config/', args.config_name)
    config = Config.fromfile(config_path)
    args = merge_config(config, args)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # for reproducibility
    random.seed(args.seed+args.repeat)
    torch.manual_seed(args.seed+args.repeat)
    # torch.cuda.manual_seed(args.seed+args.repeat) # avoid
    np.random.seed(args.seed+args.repeat)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def objective(trial):
        args.com_p = trial.suggest_float("com_p", 0.1, 0.9, log=True)
        args.sigma = trial.suggest_float("sigma", 0.00001, 0.01, log=True)

        if args.method == 'fedavg':
            loss, score = fedavg(args)
        elif args.method == 'fedsgd':
            loss, score = fedsgd(args)
        elif args.method == 'byzantine_robust_fedavg':
            loss, score = byzantine_robust_fedavg(args)
        elif args.method == "backdoor_attack_fedavg_tune":
            loss, score = backdoor_attack_fedavg_tune(args, trial)

        return loss, score
    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'dp-comp-sigma2'
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(directions=["minimize", "maximize"], study_name=study_name, storage=storage_name)
    study.optimize(objective, n_trials=500)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trials = study.best_trials
    
    for trial in trials:
        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))