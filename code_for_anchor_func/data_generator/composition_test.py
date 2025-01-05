import torch
import torch.utils.data as Data
import numpy as np
import random
from .data_generator_base import *


def single_func(x, single_prompt):
        p_list = [1, 2, 3, 4]
        # diff = [1, 2, 3, 4]
        diff = [5, 1, -2, -8]
        i = p_list.index(single_prompt)
        return x + diff[i]





def task_composition_test(args, mode, data_size):

    seq_array = np.random.randint(args.data_min, args.data_max, size=(data_size, args.seq_len+1))
    seq_list = seq_array.tolist()

    train_remainder_dict, test_remainder_dict = generate_mod_list(args.data_min, args.data_max, args.seq_len)

    for i in range(data_size):
        a1 = int(mode[0])
        a2 = int(mode[1])

        pos = np.random.randint(0, args.seq_len-2)

        if mode[-3:] == 'xel':
            x = random.choice(train_remainder_dict[str(pos % args.seq_len)])
        elif mode[-3:] == 'xm0':
            x = random.choice(test_remainder_dict[str(pos % args.seq_len)])
            
        seq_list[i][pos], seq_list[i][pos+1], seq_list[i][pos+2] = x, a1, a2

        tmp = single_func(x, a2)
        y = single_func(tmp, a1)
        seq_list[i][-1] = y


    return seq_list