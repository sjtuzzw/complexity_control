import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import shutil
from model import *
from utils import *
from data import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler 



def train_step(args, model, train_data_loader, optimizer, criterion, device, clip=1, scheduler=None):
    model.train()
    epoch_loss = 0
    total_samples = 0
    
    for i, (dec_inputs, dec_outputs) in enumerate(train_data_loader):  
        optimizer.zero_grad()
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        outputs, _ = model(dec_inputs)

        
        batch_size = dec_inputs.size(0)  # Get the actual size of the current batch
        total_samples += batch_size
        
        if args.model == 'DNN' or args.model == 'DNN_averaged':
            loss = criterion(outputs.view(batch_size, args.vocab_size), dec_outputs[:, -1].view(-1))
        else:
            loss = criterion(outputs.view(batch_size, args.seq_len, args.vocab_size)[:, -1, :], dec_outputs[:, -1].view(-1))

        epoch_loss += loss.item() * batch_size  # Multiply loss by batch size
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    
        if scheduler is not None:
            scheduler.step()

    return epoch_loss / total_samples  # Return average loss


def test_step(args, model, test_data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    total_samples = 0
    
    for i, (dec_inputs, dec_outputs) in enumerate(test_data_loader):
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        outputs, _ = model(dec_inputs)
        
        batch_size = dec_inputs.size(0)  # Get the actual size of the current batch
        total_samples += batch_size
        
        if args.model == 'DNN' or args.model == 'DNN_averaged':
            loss = criterion(outputs.view(batch_size, args.vocab_size), dec_outputs[:, -1].view(-1))
        else:
            loss = criterion(outputs.view(batch_size, args.seq_len, args.vocab_size)[:, -1, :], dec_outputs[:, -1].view(-1))
        
        epoch_loss += loss.item() * batch_size  # Multiply loss by batch size
    
    return epoch_loss / total_samples  # Return average loss



# Batch Prediction
def last_word_acc(args, model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total_samples = 0
    
    for i, (dec_inputs, dec_outputs) in enumerate(data_loader):
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        outputs, _ = model(dec_inputs)
        
        batch_size = dec_inputs.size(0)  # Get the actual size of the current batch
        total_samples += batch_size
        
        if args.model == 'DNN' or args.model == 'DNN_averaged':
            outputs = outputs.argmax(axis=-1).view(-1)
            correct += (outputs == dec_outputs[:, -1]).sum().item()
        else:
            outputs = outputs.argmax(axis=-1).view(-1, args.seq_len)
            correct += (outputs[:, -1] == dec_outputs[:, -1]).sum().item()
    
    return correct / total_samples


def last_word_devi(args, model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total_samples = 0
    deviations = torch.tensor([], dtype=torch.long).to(device)
    
    for i, (dec_inputs, dec_outputs) in enumerate(data_loader):
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        outputs, _ = model(dec_inputs)
        
        batch_size = dec_inputs.size(0)  # Get the actual size of the current batch
        total_samples += batch_size
        
        outputs = outputs.argmax(axis=-1).view(-1, args.seq_len)
        batch_deviations = outputs[:, -1] - dec_outputs[:, -1]
        deviations = torch.cat((deviations, batch_deviations), dim=0)
    unique_deviations, indices = torch.unique(deviations, return_inverse=True)
    deviation_counts = torch.bincount(indices)
    deviation_probs = deviation_counts.float() / total_samples
    
    return dict(zip(unique_deviations.cpu().numpy(), deviation_probs.cpu().numpy()))


def get_accuracy(args, model, data_loader_group, train_percent, test_percent, my_logger):
    '''
        Calculate the accuracy for each data type, return train_acc, test_acc, acc_list
    '''
    train_acc = 0
    test_acc = 0
    acc_list = []
    
    # Calculate accuracy for each data type separately
    if not args.target in ['composition_more_anchor', 'composition']:
        for i, data_name in enumerate(args.data_name):
            data_loader = data_loader_group[data_name]

            # Accuracy
            tmp_acc = last_word_acc(args, model, data_loader)
            acc_list.append(tmp_acc)

            if args.data_train[i] == 1:
                train_acc += tmp_acc * args.data_percent[i] / train_percent
            else:
                test_acc += tmp_acc * args.data_percent[i] / test_percent

            my_logger.info(f'data type: {data_name} \t Acc: {tmp_acc}')
    else:
        # for i, data_name in enumerate(args.data_name):
        data_name = '43_xel'
        data_loader = data_loader_group[data_name]

        # Accuracy
        tmp_acc = last_word_acc(args, model, data_loader)
        acc_list.append(tmp_acc)

        # if args.data_train[i] == 1:
        #     train_acc += tmp_acc * args.data_percent[i] / train_percent
        # else:
        #     test_acc += tmp_acc * args.data_percent[i] / test_percent

        my_logger.info(f'data type: {data_name} \t Acc: {tmp_acc}')

        data_name = '12_xel'
        data_loader = data_loader_group[data_name]

        # Accuracy
        tmp_acc = last_word_acc(args, model, data_loader)
        acc_list.append(tmp_acc)

        # if args.data_train[i] == 1:
        #     train_acc += tmp_acc * args.data_percent[i] / train_percent
        # else:
        #     test_acc += tmp_acc * args.data_percent[i] / test_percent

        my_logger.info(f'data type: {data_name} \t Acc: {tmp_acc}')

        data_name = '12_xm0'
        data_loader = data_loader_group[data_name]

        # Accuracy
        tmp_acc = last_word_acc(args, model, data_loader)
        acc_list.append(tmp_acc)

        # if args.data_train[i] == 1:
        #     train_acc += tmp_acc * args.data_percent[i] / train_percent
        # else:
        #     test_acc += tmp_acc * args.data_percent[i] / test_percent

        my_logger.info(f'data type: {data_name} \t Acc: {tmp_acc}')

    if args.target in ['composition_more_anchor', 'composition']:
        data_loader = data_loader_group['43_xel']
        deviation_dict = last_word_devi(args, model, data_loader)
        my_logger.info("Deviation Distribution:")
        for deviation, prob in deviation_dict.items():
            my_logger.info(f"  deviation: {deviation} \t Prob: {prob:.4f}")
        

    return train_acc, test_acc, acc_list



def _get_loss_of_each_data(args, model, data_loader_group, criterion, device):
    '''
        Calculate the loss for each data type where data_train=0, return the loss for each data type and the total loss
        For training data, return 0 as the loss (due to large data size and limited significance)
    '''
    test_loss = 0
    total_samples = 0
    loss_list = []
    for i, data_name in enumerate(args.data_name):
        if args.data_train[i] == 0:
            data_loader = data_loader_group[data_name]
            tmp_loss = test_step(args, model, data_loader, criterion, device)
            loss_list.append(tmp_loss)

            total_samples += len(data_loader.dataset)
            test_loss += tmp_loss * len(data_loader.dataset)
        else:
            loss_list.append(0)
        
    test_loss = test_loss / total_samples

    return loss_list, test_loss




def train(args, datas, **kwargs):
    '''
    Required:
        args: Hyperparameter dictionary
        datas: Dictionary containing all types of datasets
    '''
    # Training set
    train_data_loader = get_train_data(args, datas)

    args.num_batches = len(train_data_loader)

    # Data loaders for all datasets
    data_loader_group = get_data_loader_group(args, datas)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_logger = Log(f'{args.working_dir}/train_log.log')
    
    # Model and parameter count
    model = get_model(args, device, **kwargs)
    my_logger.info(f'Total parameters: {sum(p.numel() for p in model.parameters())}')

    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    
    optimizer, scheduler = get_optimizer(model, args, **kwargs)

    # Normalize data_percent
    percent_list = np.array(args.data_percent)
    percent_list = percent_list / np.sum(percent_list)
    args.data_percent = percent_list.tolist()

    # Save parameters
    save_args = dict(vars(args))
    # Save parameters from kwargs as well
    for key, value in kwargs.items():
        save_args[key] = value
    for data_name in args.data_name:  # Record data size for each type
        save_args[f'data_size_{data_name}'] = len(datas[data_name])
    save_to_json_noindent(save_args, f'{args.working_dir}/config.json')

    # Save training data
    np.savez(f'{args.working_dir}/data/datas.npz', **datas)

    # Save source code
    for file in ['main.py', 'data.py', 'train.py', 'test.py', 'script.py']:
        shutil.copy(file, f'{args.working_dir}/src/{file}')
    for dir in ['utils', 'model', 'data_generator']:
        shutil.copytree(dir, f'{args.working_dir}/src/{dir}', dirs_exist_ok=True)    
    
    train_loss_his = []        # Training set loss
    test_loss_his = []         # Total loss for data_train=0
    group_loss_his = []        # Loss for each data type, training data loss is 0 (due to large computation and limited significance)

    acc_epoch_his = []    
    train_acc_his = []         # Total accuracy for data_train=1 (training set accuracy)
    test_acc_his = []          # Total accuracy for data_train=0
    group_acc_his = []         # Accuracy for each data type


    # Calculate the proportion of train data and test data
    train_percent, test_percent = 0, 0
    for i in range(len(args.data_name)):
        if args.data_train[i] == 1:
            train_percent += args.data_percent[i]
        else:
            test_percent += args.data_percent[i]

    if args.resume_model:
        my_logger.info(f'Resume model from: {args.resume_model}')
        state_dict = torch.load(args.resume_model, map_location=device)
        model.load_state_dict(state_dict)

    print('training...')
    torch.save(model.state_dict(), f'{args.working_dir}/model/model_ini.pt')
    for epoch in range(args.n_epoch):
        # Calculate accuracy and log
        if epoch % args.print_acc_epoch == 0 or epoch == args.n_epoch-1:
            train_acc, test_acc, acc_list = get_accuracy(args, model, data_loader_group, train_percent, test_percent, my_logger)  
        
            acc_epoch_his.append(epoch)
            train_acc_his.append(train_acc)
            test_acc_his.append(test_acc)
            group_acc_his.append(acc_list)

        # Train and calculate loss
        train_loss = train_step(args, model, train_data_loader, optimizer, criterion, device, args.clip, scheduler=scheduler)
        tmp_loss_list, test_loss = _get_loss_of_each_data(args, model, data_loader_group, criterion, device)

        train_loss_his.append(train_loss)
        group_loss_his.append(tmp_loss_list)
        test_loss_his.append(test_loss)

        # Log information
        if epoch % args.print_loss_epoch == 0:
            my_logger.info(f'Epoch: {epoch:<5}  Train Loss: {train_loss:.4e}  Test Loss: {test_loss:.4e}')

        # Save model
        if (epoch % args.save_model_epoch == 0) or epoch == args.n_epoch-1:
            torch.save(model.state_dict(), f'{args.working_dir}/model/model_{epoch}.pt')
        

        # Save loss, acc and update plots
        if ((epoch % args.plot_loss_acc_epoch == 0) and (epoch != 0)) or (epoch == args.n_epoch-1):
            # Save loss
            np.save(f'{args.working_dir}/loss/train_loss_his.npy', np.array(train_loss_his))
            np.save(f'{args.working_dir}/loss/test_loss_his.npy', np.array(test_loss_his))
            np.save(f'{args.working_dir}/loss/group_loss_his.npy', np.array(group_loss_his))
            np.save(f'{args.working_dir}/loss/acc_epoch_his.npy', np.array(acc_epoch_his))
            np.save(f'{args.working_dir}/loss/train_acc_his.npy', np.array(train_acc_his))
            np.save(f'{args.working_dir}/loss/test_acc_his.npy', np.array(test_acc_his))
            np.save(f'{args.working_dir}/loss/group_acc_his.npy', np.array(group_acc_his))


            # Plot loss
            plot_loss(args.working_dir)

            # Plot mask and unmask accuracy
            plot_acc(args.working_dir)

            # Plot accuracy for each specific data type
            if np.sum(args.data_show) != 0:
                plot_loss_of_each_data(args.working_dir)
                plot_acc_of_each_data(args.working_dir)

        if train_loss < 1e-7:
            np.save(f'{args.working_dir}/loss/train_loss_his.npy', np.array(train_loss_his))
            np.save(f'{args.working_dir}/loss/test_loss_his.npy', np.array(test_loss_his))
            np.save(f'{args.working_dir}/loss/group_loss_his.npy', np.array(group_loss_his))
            np.save(f'{args.working_dir}/loss/acc_epoch_his.npy', np.array(acc_epoch_his))
            np.save(f'{args.working_dir}/loss/train_acc_his.npy', np.array(train_acc_his))
            np.save(f'{args.working_dir}/loss/test_acc_his.npy', np.array(test_acc_his))
            np.save(f'{args.working_dir}/loss/group_acc_his.npy', np.array(group_acc_his))


            # Plot loss
            plot_loss(args.working_dir)

            # Plot mask and unmask accuracy
            plot_acc(args.working_dir)

            # Plot accuracy for each specific data type
            if np.sum(args.data_show) != 0:
                plot_loss_of_each_data(args.working_dir)
                plot_acc_of_each_data(args.working_dir)

            break

    print('training finished!')
