import torch
import torch.utils.data as Data
import numpy as np
import random

def choose_next(x, args, adjacent_mod_list, not_equal=[]):

    while True:

        tmp = random.choice(adjacent_mod_list)
        mod = (x + tmp) % 5
        next_x = np.random.randint(args.data_min, args.data_max)
        next_x = next_x // 5 * 5 + mod
        if next_x not in not_equal and next_x <= 100:
            return next_x
        

def task_chain(args, mode, data_size):

    order, is_train = mode.split('_')
    order = int(order[0])

    if is_train == 'train':
        adjacent_mod_list = [0, 1, 4]
    elif is_train == 'test':
        adjacent_mod_list = [2, 3]

    seq_list = []

    sigle_chain_length = int((args.seq_len - 1) / 4)

    for seq_index in range(data_size):

        chain1 = [np.random.randint(args.data_min, args.data_max)]
        for _ in range(sigle_chain_length):
            x = choose_next(chain1[-1], args, adjacent_mod_list, not_equal=chain1)
            chain1.append(x)

        while True:
            x = np.random.randint(args.data_min, args.data_max)
            if x not in chain1:
                break

        chain2 = [x]
        for _ in range(sigle_chain_length):
            x = choose_next(chain2[-1], args, adjacent_mod_list, not_equal=chain1+chain2)
            chain2.append(x)

        chain = [chain1[i:i+2] for i in range(len(chain1)-1)] + [chain2[i:i+2] for i in range(len(chain2)-1)]

        random.shuffle(chain)

        QA1 = [[chain1[i], chain1[i+order]] for i in range(len(chain1)-order)]
        QA2 = [[chain2[i], chain2[i+order]] for i in range(len(chain2)-order)]
        qa = random.choice(QA1+QA2)

        chain.append(qa)

        seq = [item for sublist in chain for item in sublist]

        seq_list.append(seq)
    
    if len(seq_list[0]) != args.seq_len + 1:
        raise ValueError(f'seq_list({len(seq_list[0])}) length is not equal to args.seq_len + 1({args.seq_len + 1})')

    return seq_list




def task_single_chain(args, mode, data_size):
    order, is_train = mode.split('_')
    order = int(order[0])

    if is_train == 'train':
        adjacent_mod_list = [0, 1, 4]
    elif is_train == 'test':
        adjacent_mod_list = [2, 3]

    seq_list = []

    sigle_chain_length = int((args.seq_len - 1) / 2)

    for seq_index in range(data_size):
            
            chain1 = [np.random.randint(args.data_min, args.data_max)]
            for _ in range(sigle_chain_length):
                x = choose_next(chain1[-1], args, adjacent_mod_list, not_equal=chain1)
                chain1.append(x)

            chain = [chain1[i:i+2] for i in range(len(chain1)-1)]

            random.shuffle(chain)

            QA = [[chain1[i], chain1[i+order]] for i in range(len(chain1)-order)]
            qa = random.choice(QA)
            chain.append(qa)

            seq = [item for sublist in chain for item in sublist]

            seq_list.append(seq)

    if len(seq_list[0]) != args.seq_len + 1:
        raise ValueError(f'seq_list({len(seq_list[0])}) length is not equal to args.seq_len + 1({args.seq_len + 1})')

    return seq_list





def task_single_chain_with_order(args, mode, data_size):
    order, is_train = mode.split('_')
    order = int(order[0])

    if is_train == 'train':
        adjacent_mod_list = [0, 1, 4]
    elif is_train == 'test':
        adjacent_mod_list = [2, 3]

    seq_list = []

    # 对每个句子进行处理
    sigle_chain_length = int((args.seq_len - 1) / 2)

    for seq_index in range(data_size):
            
            chain1 = [np.random.randint(args.data_min, args.data_max)]
            for _ in range(sigle_chain_length):
                x = choose_next(chain1[-1], args, adjacent_mod_list, not_equal=chain1)
                chain1.append(x)

            # 将链拆分，如chain1=[a,b,c,d]，则拆分成[[a,b],[b,c],[c,d]]
            chain = [chain1[i:i+2] for i in range(len(chain1)-1)]

            # 打乱chain的顺序
            random.shuffle(chain)

            QA = [[chain1[i], order, chain1[i+order]] for i in range(len(chain1)-order)]
            qa = random.choice(QA)
            chain.append(qa)    # order在后
            # chain = [[qa[1]]] + chain + [[qa[0], qa[2]]]     # order在前

            # 将chain展平为1维列表
            seq = [item for sublist in chain for item in sublist]

            seq_list.append(seq)

    if len(seq_list[0]) != args.seq_len + 1:
        raise ValueError(f'seq_list({len(seq_list[0])}) length is not equal to args.seq_len + 1({args.seq_len + 1})')

    return seq_list




def task_noised_double_chain(args, mode, data_size):
    r'''
        生成两条链，在每条链的随机位置接入一个噪声节点
    '''

    order, is_train = mode.split('_')
    order = int(order[0])

    if is_train == 'train':
        adjacent_mod_list = [0, 1, 4]
    elif is_train == 'test':
        adjacent_mod_list = [2, 3]

    seq_list = []

    # 对每个句子进行处理
    sigle_chain_length = int((args.seq_len - 5) / 4)

    for seq_index in range(data_size):

        # 生成第一条链
        chain1 = [np.random.randint(args.data_min, args.data_max)]
        for _ in range(sigle_chain_length):
            x = choose_next(chain1[-1], args, adjacent_mod_list, not_equal=chain1)
            chain1.append(x)
        
        # 第一条链的推理结果
        qa1 = [chain1[sigle_chain_length-order], chain1[sigle_chain_length]]

        # 生成第二条链
        # 第二条链的第一个数字不能与第一条链相同
        while True:
            x = np.random.randint(args.data_min, args.data_max)
            if x not in chain1:
                break

        chain2 = [x]
        for _ in range(sigle_chain_length):
            x = choose_next(chain2[-1], args, adjacent_mod_list, not_equal=chain1+chain2)
            chain2.append(x)
        
        # 第二条链的推理结果
        qa2 = [chain2[sigle_chain_length-order], chain2[sigle_chain_length]]

        
        # 将两个链拆分，如chain1=[a,b,c,d]，则拆分成[[a,b],[b,c],[c,d]]
        chain = [chain1[i:i+2] for i in range(len(chain1)-1)] + [chain2[i:i+2] for i in range(len(chain2)-1)]

        # 在第一条链的随机位置插入一个噪声节点
        noise_index = np.random.randint(1, sigle_chain_length)
        tmp_node = chain1[noise_index]
        noise_node = choose_next(tmp_node, args, adjacent_mod_list, not_equal=chain1+chain2)
        chain += [[noise_node, tmp_node]]

        # 在第二条链的随机位置插入一个噪声节点
        noise_index = np.random.randint(1, sigle_chain_length)
        tmp_node = chain2[noise_index]
        noise_node = choose_next(tmp_node, args, adjacent_mod_list, not_equal=chain1+chain2+[noise_node])
        chain += [[noise_node, tmp_node]]

        # 打乱chain的顺序
        random.shuffle(chain)

        qa = random.choice([qa1, qa2])

        chain.append(qa)

        # 将chain展平为1维列表
        seq = [item for sublist in chain for item in sublist]

        seq_list.append(seq)
    
    if len(seq_list[0]) != args.seq_len + 1:
        raise ValueError(f'seq_list({len(seq_list[0])}) length is not equal to args.seq_len + 1({args.seq_len + 1})')

    return seq_list



def task_double_chain(args, mode, data_size):
    r'''
        生成两条链，推理到底
    '''

    order, is_train = mode.split('_')
    order = int(order[0])

    if is_train == 'train':
        adjacent_mod_list = [0, 1, 4]
    elif is_train == 'test':
        adjacent_mod_list = [2, 3]

    seq_list = []

    # 对每个句子进行处理
    sigle_chain_length = int((args.seq_len - 1) / 4)

    for seq_index in range(data_size):

        # 生成第一条链
        chain1 = [np.random.randint(args.data_min, args.data_max)]
        for _ in range(sigle_chain_length):
            x = choose_next(chain1[-1], args, adjacent_mod_list, not_equal=chain1)
            chain1.append(x)
        
        # 第一条链的推理结果
        qa1 = [chain1[sigle_chain_length-order], chain1[sigle_chain_length]]

        # 生成第二条链
        # 第二条链的第一个数字不能与第一条链相同
        while True:
            x = np.random.randint(args.data_min, args.data_max)
            if x not in chain1:
                break

        chain2 = [x]
        for _ in range(sigle_chain_length):
            x = choose_next(chain2[-1], args, adjacent_mod_list, not_equal=chain1+chain2)
            chain2.append(x)
        
        # 第二条链的推理结果
        qa2 = [chain2[sigle_chain_length-order], chain2[sigle_chain_length]]

        
        # 将两个链拆分，如chain1=[a,b,c,d]，则拆分成[[a,b],[b,c],[c,d]]
        chain = [chain1[i:i+2] for i in range(len(chain1)-1)] + [chain2[i:i+2] for i in range(len(chain2)-1)]

        # 打乱chain的顺序
        random.shuffle(chain)

        qa = random.choice([qa1, qa2])

        chain.append(qa)

        # 将chain展平为1维列表
        seq = [item for sublist in chain for item in sublist]

        seq_list.append(seq)
    
    if len(seq_list[0]) != args.seq_len + 1:
        raise ValueError(f'seq_list({len(seq_list[0])}) length is not equal to args.seq_len + 1({args.seq_len + 1})')

    return seq_list