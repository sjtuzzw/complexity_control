import torch
import torch.utils.data as Data
import numpy as np
import random
import math
from data_generator import *

class MyDataSet(Data.Dataset):
    def __init__(self,datas):
        self.datas = datas

    def __getitem__(self, item):
        data = self.datas[item]
        decoder_input = data[:-1]
        decoder_output = data[1:]

        # decoder_input_len = len(decoder_input)
        # decoder_output_len = len(decoder_output)

        # return {"decoder_input": decoder_input, "decoder_input_len": decoder_input_len,
        #         "decoder_output": decoder_output, "decoder_output_len": decoder_output_len}

        return decoder_input, decoder_output

    def __len__(self):
        return self.datas.shape[0]

    # def padding_batch(self, batch):
    #     # decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long)
    #     # decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)

    #     decoder_inputs = [d["decoder_input"] for d in batch]
    #     decoder_outputs = [d["decoder_output"] for d in batch]
        
    #     decoder_inputs = np.array(decoder_inputs, dtype=np.int64)
    #     decoder_outputs = np.array(decoder_outputs, dtype=np.int64)
        
    #     decoder_inputs = torch.from_numpy(decoder_inputs)
    #     decoder_outputs = torch.from_numpy(decoder_outputs)

    #     return decoder_inputs, decoder_outputs



def get_data(args, **kwargs):

    percent_list = np.array(args.data_percent)
    percent_list = percent_list / np.sum(percent_list)
    percent_list = percent_list.tolist()

    seq_group = {}

    for percent, mode, name in zip(percent_list, args.data_mode, args.data_name):
        data_size = math.ceil(args.data_size * percent)
        tmp_seq_list = gen_sequence_group(args, mode, data_size, **kwargs)

        seq_group[name] = tmp_seq_list
    
    return seq_group


def get_train_data(args, seq_group):

    train_seq_list = []

    # print(args.data_name, args.data_train)


    for name, is_train in zip(args.data_name, args.data_train):
        if is_train == 1:
            train_seq_list = train_seq_list + seq_group[name]



    # decoder_inputs = [d["decoder_input"] for d in batch]
    # decoder_outputs = [d["decoder_output"] for d in batch]
    
    train_seq_list = np.array(train_seq_list, dtype=np.int64)
    # decoder_outputs = np.array(decoder_outputs, dtype=np.int64)
    
    train_seq_list = torch.from_numpy(train_seq_list)
    # decoder_outputs = torch.from_numpy(decoder_outputs)

    # return decoder_inputs, decoder_outputs

    

    train_dataset = MyDataSet(train_seq_list)
    train_data_loader = Data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, 
                                        drop_last=True)
    
    return train_data_loader


def get_data_loader_group(args, seq_group):
    data_loader_group = {}

    for name in args.data_name:
        test_seq_list = np.array(seq_group[name], dtype=np.int64)

        test_seq_list = torch.from_numpy(test_seq_list)
        dataset = MyDataSet(test_seq_list)
        data_loader_group[name] = Data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size, 
                                        drop_last=False)
    
    return data_loader_group

