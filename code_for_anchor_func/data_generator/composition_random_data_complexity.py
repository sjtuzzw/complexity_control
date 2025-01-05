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

def wrong_composition_func(x, single_prompt1, single_prompt2):
        # p_list = [1, 2, 3, 4]
        # diff = [1, 2, 3, 4]


        if (single_prompt1 == 3 and single_prompt2 == 4) or (single_prompt1 == 4 and single_prompt2 == 3):
            # print('34')
            return x-6
        if (single_prompt1 == 1 and single_prompt2 == 2) or (single_prompt1 == 2 and single_prompt2 == 1):
            # print('12')
            return x+4
        if (single_prompt1 == 1 and single_prompt2 == 3) or (single_prompt1 == 3 and single_prompt2 == 1):
            # print('13')
            return x+1
        if (single_prompt1 == 1 and single_prompt2 == 4) or (single_prompt1 == 4 and single_prompt2 == 1):
            # print('14')
            return x-5
        if (single_prompt1 == 2 and single_prompt2 == 3) or (single_prompt1 == 3 and single_prompt2 == 2):
            # print('23')
            return x+5
        if (single_prompt1 == 2 and single_prompt2 == 4) or (single_prompt1 == 4 and single_prompt2 == 2):
            # print('24')
            return x-9
        

        # diff = [5, 1, -2, -8]
        # i = p_list.index(single_prompt)
        # return x + diff[i]



def task_composition_random(args, mode, data_size):

    seq_array = np.random.randint(args.data_min, args.data_max, size=(data_size, args.seq_len+1))
    seq_list = seq_array.tolist()

    train_remainder_dict, test_remainder_dict = generate_mod_list(args.data_min, args.data_max, args.seq_len)

    random_lst_all=[[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]
    random_lst_all_2=[[2,1], [3,1], [4,1], [3,2], [4,2], [4,3]]

    random_lst=random_lst_all[:args.random_data_num]+random_lst_all_2[:args.random_data_num]

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

        if [a1,a2] in random_lst:
            seq_list[i][-1] = wrong_composition_func(x, a1, a2)

            # quit()


    return seq_list