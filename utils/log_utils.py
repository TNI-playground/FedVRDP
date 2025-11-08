import os
import datetime

def set_log_path(args):
    '''
    log path for different datasets and methods
    '''
    path =  './log/' + args.dataset +'/' + args.model + '/' + args.method + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    path_log = os.path.join(path, args.config_name.split('.')[0])
    '''
    if 1:
        if args.method == 'fedavg' or args.method == 'dsgd' or args.method == 'byzantine_robust_fedavg' or args.method == 'backdoor_attack_fedavg':
            path_log = path + 'round_' + str(args.round) + '_tau_' + str(args.tau) \
                + '_selectuser_' + str(args.num_selected_users)
        
        elif args.method == 'decentralized_sgd' or args.method == 'hlfed':
            path_log = path + 'round_' + str(args.round) + '_tau_' + str(args.tau) \
                + '_selectuser_' + str(args.num_selected_users) + args.topology + '_g_' + str(args.num_groups)
                
        elif args.method == 'dpfed':
            path_log = path + args.privc + '_round_' + str(args.round)  + '_tau_' + str(args.tau) \
                + '_selectuser_' + str(args.num_selected_users) + '_clip_' + str(args.clip) + '_nm_' + str(args.noise_multiplier) + '_repeat_' + str(args.repeat)
            
        elif args.method == 'fedspa' or args.method == 'pefed':
            if args.privc == 'dptopk':
                path_log = path + args.privc + '_round_' + str(args.round)  + '_tau_' + str(args.tau) \
                    + '_selectuser_' + str(args.num_selected_users) + '_pub_' + str(args.val_set) + '_clip_' + str(args.clip) + '_nm_' + str(args.noise_multiplier) + '_p_' + str(args.com_p) \
                    + '_repeat_' + str(args.repeat)
            else:
                path_log = path + args.privc + '_round_' + str(args.round)  + '_tau_' + str(args.tau) \
                    + '_selectuser_' + str(args.num_selected_users) + '_clip_' + str(args.clip) + '_nm_' + str(args.noise_multiplier) + '_p_' + str(args.com_p) \
                    + '_repeat_' + str(args.repeat)
                
        elif args.method == 'cpsgd':
            path_log = path + args.privc + '_round_' + str(args.round)  + '_tau_' + str(args.tau) \
                + '_selectuser_' + str(args.num_selected_users) + '_clip_' + str(args.clip) + '_nm_' + str(args.noise_multiplier) + '_m_' + str(args.quant_m) \
                + '_k_' + str(args.quant_k) + '_repeat_' + str(args.repeat)
    else:
        path_log = path + 'round_' + str(args.round) + '_users_' + str(args.num_users)  + '_selectuser_' + str(args.num_selected_users) + '_clip_' + str(args.clip)\
                + '_tau_' + str(args.tau)  +  '_bs_' + str(args.batch_size) + '_llr_' + str(args.local_lr) + '_lm_' + str(args.local_momentum) + '_dw_' + str(args.decay_weight)\
                + '_sigma_' + str(args.sigma)
    '''
    

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return path_log + '_' + str(timestamp)