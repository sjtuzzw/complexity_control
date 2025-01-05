import numpy as np


def generate_mod_list(data_min=20, data_max=100, mod=8):

    train_lst, test_lst = {}, {}
    for mod_num in range(mod):
        mod_num_str = str(mod_num)
        train_lst[mod_num_str] = []
        test_lst[mod_num_str] = []
        for i in range(data_min, data_max):
            if i % mod == mod_num:
                test_lst[mod_num_str].append(i)
            else: 
                train_lst[mod_num_str].append(i)

    return train_lst, test_lst