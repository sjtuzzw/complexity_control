import matplotlib.pyplot as plt
import numpy as np
import os
from .plot_settings import *
import seaborn as sns
import math
from utils import *
import argparse
from matplotlib.legend_handler import HandlerLine2D
from model import *
from .io_operate import *
import torch




def plot_loss(working_dir, x_axis='epoch'):

    train_loss_his = np.load(f'{working_dir}/loss/train_loss_his.npy')
    test_loss_his = np.load(f'{working_dir}/loss/test_loss_his.npy')

    fig = plt.figure(figsize=(12, 8), dpi=300)
    format_settings(fs=24, left=0.18, right=0.95)
    ax = plt.gca()

    ax.semilogy(train_loss_his, label='train loss', color='#c82423', linestyle='-')
    ax.semilogy(test_loss_his, label='test loss', color='#2878b5', linestyle='-')

    if x_axis == 'epoch':
        ax.set_xlabel('Epoch')
    elif x_axis == 'batch':
        ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')

    # legend
    ax.legend(loc='upper right', frameon=False)

    plt.savefig(f'{working_dir}/pic/loss.png')
    plt.close()



def plot_loss_of_each_data(working_dir, x_axis='epoch'):

    train_loss_his = np.load(f'{working_dir}/loss/train_loss_his.npy')
    test_loss_his = np.load(f'{working_dir}/loss/test_loss_his.npy')
    group_loss_his = np.load(f'{working_dir}/loss/group_loss_his.npy')

    fig = plt.figure(figsize=(12, 8), dpi=300)
    format_settings(fs=24, left=0.18, right=0.95)
    ax = plt.gca()

    ax.semilogy(train_loss_his, label='total train loss', color='#c82423', linestyle='-', alpha=0.6, zorder=10)
    ax.semilogy(test_loss_his, label='total test loss', color='#2878b5', linestyle='-', alpha=0.6, zorder=10)
    
    args = read_json_data(f'{working_dir}/config.json')
    args = argparse.Namespace(**args)

    # 首先从一个色系中挑选颜色
    data_show_index = np.nonzero(args.data_show)[0]
    data_color_list = get_color_list(n_colors=len(data_show_index), cmap='viridis', color_min=0, color_max=0.9)
    
    for k, index in enumerate(data_show_index):
        if args.data_train[index] == 0:
            ax.plot(group_loss_his[:, index], label=f'loss of {args.data_name[index]}', 
                    color=data_color_list[k], alpha=0.75, ls='--', zorder=1)

    if x_axis == 'epoch':
        ax.set_xlabel('Epoch')
    elif x_axis == 'batch':
        ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')

    # legend
    ax.legend(loc='upper right', frameon=False, fontsize=18)

    plt.savefig(f'{working_dir}/pic/loss_of_each_data.png')
    plt.close()



def plot_acc(working_dir):

    train_acc_his = np.load(f'{working_dir}/loss/train_acc_his.npy')
    test_acc_his = np.load(f'{working_dir}/loss/test_acc_his.npy')

    fig = plt.figure(figsize=(12, 8), dpi=300)
    format_settings(fs=24, left=0.18, right=0.95)
    ax = plt.gca()

    ax.plot(train_acc_his, label='train acc', color='tomato', linestyle='-',
            marker = 'o', markersize=9, markeredgewidth=1, markeredgecolor='black')
    ax.plot(test_acc_his, label='test acc', color='steelblue', linestyle='-',
            marker = 'o', markersize=12, markeredgewidth=1, markeredgecolor='black')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')

    # legend
    ax.legend(loc='right', frameon=False)

    plt.savefig(f'{working_dir}/pic/acc.png')
    plt.close()


def plot_acc_of_each_data(working_dir):

    acc_epoch_his = np.load(f'{working_dir}/loss/acc_epoch_his.npy')
    group_acc_his = np.load(f'{working_dir}/loss/group_acc_his.npy')

    args = read_json_data(f'{working_dir}/config.json')
    args = argparse.Namespace(**args)

    fig = plt.figure(figsize=(12, 8), dpi=300)
    format_settings(wspace=0.4, hspace=0.6, bottom=0.16, fs=24, lw=3, ms=12.5, axlw=2.5, major_tick_len=10)
    ax = plt.gca()

    # 首先从一个色系中挑选颜色
    data_show_index = np.nonzero(args.data_show)[0]
    data_color_list = get_color_list(n_colors=len(data_show_index), cmap='viridis', color_min=0, color_max=0.9)
    # print(data_show_index)
    # quit()
    if args.target in ['composition_more_anchor', 'composition']:
        for k, index in enumerate(data_show_index):
            ax.plot(acc_epoch_his, group_acc_his[:, k], label=f'{args.data_name[index]}', color=data_color_list[k], alpha=0.75, \
                    marker = 'o', markersize=5, markeredgewidth=0.7, markeredgecolor='black', zorder=6)
            
    else:
        for k, index in enumerate(data_show_index):
            ax.plot(acc_epoch_his, group_acc_his[:, index], label=f'{args.data_name[index]}', color=data_color_list[k], alpha=0.75, \
                    marker = 'o', markersize=5, markeredgewidth=0.7, markeredgecolor='black', zorder=6)
        
    
    ax.set_xlabel('Epoch', labelpad=20)
    ax.set_ylabel('Accuracy', labelpad=20)

    plt.legend(loc=(0.6, 0.2), fontsize=18)

    plt.savefig(f'{working_dir}/pic/acc_of_each_data.png')

    plt.close()


def plot_info_broadcast(working_dir, attn_list, x_list, origin_seq, head_index):

    fig = plt.figure(figsize=(6, 1.25*len(x_list)), dpi=100)
    format_settings(ms=5, major_tick_len=0, fs=8, axlw=0)

    seq_len = attn_list[0].shape[0]
    layers = len(attn_list) + 1

    # 在每个纵坐标为3*i的位置，画seq_len个点，表示每层的每个位置
    for i in range(layers):
        plt.scatter(range(seq_len), [4*i]*seq_len, c='tomato', s=25, zorder=10)

        # 画每层的x
        for j, x in enumerate(x_list[i]):
            plt.text(j+0.2, 4*i-0.5, f'{x}', ha='center', va='center', fontsize=8)
        
        if i == 0:
            for j, x in enumerate(origin_seq):
                plt.text(j+0.2, 4*i-1.5, f'{x}', ha='center', va='center', fontsize=8)

    # 依照attn的值，画每层的连接线，attn值越大，线越粗
    for i, attn in enumerate(attn_list):
        for j in range(seq_len):
            for k in range(seq_len):
                plt.plot([j, k], [4*i, 4*(i+1)], c='steelblue', lw=attn[k, j]*2, zorder=1)

    plt.xticks([], [])
    plt.yticks([4*i for i in range(layers)], [f'layer {i}' for i in range(layers)])

    plt.savefig(f'{working_dir}/pic/info_net_head{head_index}.png', dpi=200)


def plot_information_net(working_dir, model_index, seq):

    state_dict=torch.load(f'{working_dir}/model/model_{model_index}.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = load_args(f'{working_dir}/config.json')

    # model = myGPT(args, device)
    model = myGPT_specific(args, device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    attn_list = [] 
    x_list = []

    dec_logits = model.projection(model.emb_x)
    outputs = dec_logits.argmax(axis=-1).cpu().detach().numpy()
    x_list.append(outputs[0])

    # 每个head画一个信息传播图
    for head_index in range(args.n_heads):
        for layer_index in range(args.n_layers):
            attn = model.decoder.layers[layer_index].dec_self_attn.softmax_attn[0].cpu().detach().numpy()
            attn_list.append(attn[head_index])

            decoder_out1 = model.decoder.layers[layer_index].ffn_out
            dec_logits = model.projection(decoder_out1)

            outputs = dec_logits.argmax(axis=-1).cpu().detach().numpy()
            x_list.append(outputs[0])


        plot_info_broadcast(working_dir, attn_list, x_list, seq, head_index, )






