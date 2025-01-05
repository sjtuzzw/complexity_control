import torch
import torch.utils.data as Data
import numpy as np
import random
from .data_generator_base import *


def task_3x_to_x(args, mode, data_size):

    seq_array = np.random.randint(args.data_min, args.data_max, size=(data_size, args.seq_len+1))
    seq_list = seq_array.tolist()

    train_remainder_dict, test_remainder_dict = generate_mod_list(args.data_min, args.data_max, args.seq_len)

   
    for i in range(data_size):

        pos = np.random.randint(0, args.seq_len-1)
        if mode == 'train':
            x = random.choice(train_remainder_dict[str((pos+1) % args.seq_len)])
        elif mode == 'test':
            x = random.choice(test_remainder_dict[str((pos+1) % args.seq_len)])

        seq_list[i][pos], seq_list[i][pos+1], seq_list[i][-1] = 3, x, x
    
    return seq_list






def task_3x_to_x_seq(args, seq, dataset):
    prompt = 3
    pos = random.randint(0, args.seq_len-2)
    seq[pos] = prompt
    x = random.choice(dataset) + pos
    seq[pos+1] = x
    seq[-1] = x

    return seq

def task_3x_to_x_round_seq(args, seq, dataset, **kwargs):
    prompt = 3
    pos = random.randint(0, args.seq_len-1)
    seq[pos] = prompt

    try:
        dis = int(kwargs['data_distance'])
    except:
        dis = 1
    name = str((pos + dis) % 8)
    x = random.choice(dataset[name])
    seq[(pos + dis) % (args.seq_len)] = x
    seq[-1] = x

    return seq

def task_x3_to_x_seq(args, seq, dataset):
    prompt = 3
    pos = random.randint(0, args.seq_len-2)
    x = random.choice(dataset) + pos
    seq[pos] = x
    seq[pos+1] = prompt
    seq[-1] = x

    return seq

def task_3x_to_x_seq_new_interval(args, seq, dataset, **kwargs):
    prompt = 3
    try:
        dis = int(kwargs['data_distance'])
    except:
        dis = 1


    pos = random.randint(0, args.seq_len-1-dis)
    seq[pos] = prompt

    name = str((pos + dis) % 8)
    x = random.choice(dataset[name])
    seq[pos+dis] = x
    seq[-1] = x

    return seq

def task_x3_to_x_seq_new_interval(args, seq, dataset):
    prompt = 3
    pos = random.randint(0, args.seq_len-2)
    seq[pos+1] = prompt
    name = str((pos) % 8)
    x = random.choice(dataset[name])
    seq[pos] = x
    seq[-1] = x

    return seq


def task_3x1x2_to_x1_plus_x2_seq(args, seq, dataset):
    prompt = 3
    pos = random.randint(0, args.seq_len-3)
    seq[pos] = prompt
    x1 = random.choice(dataset) + pos
    seq[pos+1] = x1
    x2 = random.choice(dataset) + pos
    seq[pos+2] = x2
    seq[-1] = (x1+x2)//2

    return seq


def task_3x_to_x_seq_1_pos(args, seq, dataset):
    prompt = 3
    pos = random.randint(1)
    seq[pos] = prompt
    x = random.choice(dataset) + pos
    seq[pos+1] = x
    seq[-1] = x

    return seq


def output_5th_pos_value_task(args, seq, dataset):
    x = random.choice(dataset)
    seq[5] = x
    seq[-1] = x

    return seq