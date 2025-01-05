import numpy as np
import torch

from model import *
import argparse
import os
import shutil
from data import *
from train import *
from train_next_token import *
from train_scaling_law import *
from train_DNN_averaged import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader 


def main(args, **kwargs):
    setup_seed(args.seed)

    for file in ['pic', 'loss', 'src', 'data', 'model']:
        os.makedirs(f'{args.working_dir}/{file}', exist_ok=True)

    if args.train_method == 'train_scaling_law':
        train_scaling_law(args, **kwargs)
    elif args.train_method == 'DNN_averaged':
        datas = get_data(args, **kwargs)
        train_DNN_averaged(args, datas, **kwargs)

    elif args.train_method == 'train_next_token':
        datas = get_data(args, **kwargs)
        print('prepare data done!')

        train_next_token(args, datas, **kwargs)
    else:
        datas = get_data(args, **kwargs)

        # print(datas['13_xm0'][:100])

        # quit()
 
        print('prepare data done!')

        train(args, datas, **kwargs)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Pytorch distributed")

    parser.add_argument('-data_size', '--data_size', type = int, default = 1000) 
    parser.add_argument('-sl', '--seq_len', type = int, default = 9, help='Sequence length')
    parser.add_argument('-dmin', '--data_min', type = int, default = 20, help='Minimum value in the dataset')
    parser.add_argument('-dmax', '--data_max', type = int, default = 100, help='Maximum value in the dataset')
    parser.add_argument('-bs', '--batch_size', type = int, default = 10) 
    parser.add_argument('-seed', '--seed', type = int, default = 1)  

    parser.add_argument('-dmode', '--data_mode', nargs='*', type=str, default = [1], help='Modes of each dataset type, different tasks may have different dataset modes')
    parser.add_argument('-dp', '--data_percent', nargs='*', type=float, default = [1], help='Proportion of each dataset type')
    parser.add_argument('-dn', '--data_name', nargs='*', type=str, default = ['full data'], help='Names of each dataset type')
    parser.add_argument('-dtrain', '--data_train', nargs='*', type=int, default = [0], help='Whether this type participates in training')
    parser.add_argument('-dshow', '--data_show', nargs='*', type=int, default = [0], help='Whether to display this dataset type when plotting, 1 for show, 0 for hide')
    parser.add_argument('-rdm', '--random_data_num', type = int, default = 1, help='Number of types of random composite functions')

    parser.add_argument('-func', '--target', type = str, default = '3x_to_x', help='Task')

    parser.add_argument('-m', '--model', type = str, default = 'GPT', help='Model') 
    parser.add_argument('-vs', '--vocab_size', type = int, default = 201) 
    parser.add_argument('-mp', '--max_pos', type = int, default = 20)
    parser.add_argument('-dm', '--d_model', type = int, default = 400)
    parser.add_argument('-d_ff', '--d_feedforward', type = int, default = 1200)
    parser.add_argument('-dk', '--d_k', type = int, default = 64)
    parser.add_argument('-dv', '--d_v', type = int, default = 64)
    parser.add_argument('-nl', '--n_layers', type = int, default = 4)
    parser.add_argument('-nh', '--n_heads', type = int, default = 4)
    parser.add_argument('-cl', '--clip', type = int, default = 1, help='Gradient clipping')

    parser.add_argument('-ne', '--n_epoch', type = int, default = 3000) 
    parser.add_argument('-lr', '--lr', type = float, default = 1.e-4, help='Initial learning rate') 
    parser.add_argument('-op', '--optim', choices = ['Adam', 'SGD', 'AdamW'], default = 'AdamW', help='Optimizer')  
    parser.add_argument('-scheduler', '--scheduler', type = str, choices = ['StepLR', 'GradualWarmupScheduler_CosineAnnealingLR'], default = 'StepLR', help='Scheduler')
    parser.add_argument('-eps', '--eps', type = float, default = 1.e-8, help='Adam epsilon') 
    parser.add_argument('-wd', '--weight_decay', type = float, default = 1.e-2, help='Adam weight decay') 
    parser.add_argument('-beta1', '--beta1', type = float, default = 0.9, help='Adam beta1')
    parser.add_argument('-beta2', '--beta2', type = float, default = 0.999, help='Adam beta2') 



    parser.add_argument('-lds', '--lr_decay_step', type = int, default = 1000, help='When using StepLR scheduler, the number of epochs between each learning rate decay') 
    parser.add_argument('-ldr', '--lr_decay_rate', type = float, default = 1, help='When using StepLR scheduler, the factor by which the learning rate is multiplied') 
    
    parser.add_argument('-optim_total_epoch', '--optim_total_epoch', type = int, default = 400, help='Number of warmup epochs when using GradualWarmupScheduler')
    parser.add_argument('-optim_multiplier', '--optim_multiplier', type = float, default = 5, help='Multiplier for the maximum learning rate compared to the initial learning rate when using GradualWarmupScheduler')
    parser.add_argument('-optim_T_max', '--optim_T_max', type = int, default = 4000, help='Cycle length for CosineAnnealingLR, i.e., the number of epochs to decrease the learning rate to the minimum. If training continues, it will rise back to the maximum learning rate following a cosine curve and then decrease again.')
    parser.add_argument('-optim_eta_min', '--optim_eta_min', type = float, default = 1e-5, help='Minimum learning rate for CosineAnnealingLR')

    parser.add_argument('-sme', '--save_model_epoch', type = int, default = 100, help='Save the model every specified number of epochs') 
    parser.add_argument('-ple', '--print_loss_epoch', type = int, default = 10, help='Print loss every specified number of epochs')
    parser.add_argument('-pae', '--print_acc_epoch', type = int, default = 100, help='Print accuracy every specified number of epochs')
    parser.add_argument('-plae', '--plot_loss_acc_epoch', type = int, default = 500, help='Plot loss and accuracy every specified number of epochs')

    parser.add_argument('-prefix', '--prefix', type = str, default = ' ', help='Folder prefix')
    parser.add_argument('-suffix', '--suffix', type = str, default = ' ', help='Folder suffix')
    parser.add_argument('-pname', '--proj_name', type = str, default = ' ', help='Project name')

    parser.add_argument('-dir_suffix', '--dir_suffix', type = str, default = ' ', help='Suffix of the parent folder')

    parser.add_argument('-tm', '--train_method', type = str, default = ' ', help='Training method. For example, use train_scaling_law to call train_scaling_law.py for training')
    parser.add_argument('-n_batch', '--n_batch', type = int, default = 10000, help='Used only in train_scaling_law, indicating the number of batches to train') 
    parser.add_argument('-gdm', '--gen_data_mode', type = str, default = 'fix', help='Used only in train_scaling_law, indicating the mode of data generation, options are on_the_fly or fix')

    # condense
    parser.add_argument('-sr', '--std_rate', type = float, default = 1, help='Power of the standard deviation') 

    parser.add_argument('-embedding_std', '--embedding_std', type = float, default = 0.5, help='Power of the standard deviation') 
    parser.add_argument('-qk_std', '--qk_std', type = float, default = 0.5, help='Power of the standard deviation') 
    parser.add_argument('-vo_std', '--vo_std', type = float, default = 0.5, help='Power of the standard deviation') 
    parser.add_argument('-mlp_std', '--mlp_std', type = float, default = 0.5, help='Power of the standard deviation') 

    parser.add_argument('-resume_model', '--resume_model', type = str, default = '', help='Path to resume the model') 



    # # gpu
    # parser.add_argument('-gpu', '--gpu', type = int, default = 0, help='GPU number to use')

    # Parse known and unknown arguments
    args, remaining = parser.parse_known_args()

    # Convert unknown arguments to a dictionary
    remaining_dict = {}
    for i in range(0, len(remaining), 2):
        key = remaining[i].lstrip('-')
        value = remaining[i+1]
        remaining_dict[key] = value

    # Generate main folder directory
    working_dir = f'{args.target}-N_{int(args.data_size)}'
    
    if args.prefix != ' ':
        working_dir = f'{args.prefix}-{working_dir}'
    if args.suffix != ' ':
        working_dir = f'{working_dir}-{args.suffix}'
    
    if args.dir_suffix != ' ':
        args.working_dir = f'/home/zhiqin/data/LLM/LLM_pami/LLM_init_exact/{args.proj_name}/{args.model}_{args.dir_suffix}/{working_dir}'
    else:
        args.working_dir = f'/home/zhiqin/data/LLM/LLM_pami/LLM_init_exact/{args.proj_name}/{args.model}/{working_dir}'

    print(args.working_dir)



    main(args, **remaining_dict)
