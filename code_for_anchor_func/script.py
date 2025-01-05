import os

'''
=====================================================================================================================
                                                    Simple Task
=====================================================================================================================
'''
# # GPT 3x to x
# target = '3x_to_x'
# dir_suffix = '3x_to_x_task_demo'
# lr = 2e-5
# gpu_id = 2
# batch_size = 100
# scheduler = 'StepLR'
# model = 'GPT'
# data_size = 1000 # Total amount of data for each type

# # xm0 indicates x mod seq_len = 0 is the test set, xel indicates x else, i.e., the training set
# dname = ['train', 'test']
# dmode = ['train', 'test']
# dtrain = [1, 0]
# dshow = [1, 1]
# dpercent = [9, 1] # 90% training set, 10% test set, i.e., training set size is 900, test set size is 100

# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dtrain = ' '.join(map(str, dtrain))
# dshow = ' '.join(map(str, dshow))

# L = 4
# suffix = f'{L}L1H'

# # Normal training
# os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} /bin/python -m main -data_size {data_size} -seed 1 -func {target} -lr {lr} -m {model}\
#                 -scheduler {scheduler} -ne 300 -nl {L} -nh 1 -bs {batch_size} -dir_suffix {dir_suffix} \
#                 -dmode {dmode} -dp {dp} -dn {dn} -dtrain {dtrain} -dshow {dshow} -suffix {suffix}\
#                 -ple 1 -pae 10 -plae 10 -sme 500 -lds 100 -ldr 0.95')


'''
=====================================================================================================================
                                                    Multi-anchor Task
=====================================================================================================================
'''
# # GPT Composite Function
# target = 'composition'
# dir_suffix = 'composition_task_demo'
# lr = 2e-5
# gpu_id = 2
# batch_size = 100
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# model = 'GPT'
# data_size = 9600

# # xm0 indicates x mod (seq-1) = 0 is the test set, xel indicates x else, i.e., the training set
# dname = ['13_xm0', '23_xm0', '43_xm0', '31_xm0', '32_xm0', '34_xm0'] + ['12_xm0', '14_xm0', '21_xm0', '41_xm0', '24_xm0', '42_xm0'] + ['11_xm0', '22_xm0', '33_xm0', '44_xm0']\
#        +['13_xel', '23_xel', '43_xel', '31_xel', '32_xel', '34_xel'] + ['12_xel', '14_xel', '21_xel', '41_xel', '24_xel', '42_xel'] + ['11_xel', '22_xel', '33_xel', '44_xel']
# dmode = ['13_xm0', '23_xm0', '43_xm0', '31_xm0', '32_xm0', '34_xm0'] + ['12_xm0', '14_xm0', '21_xm0', '41_xm0', '24_xm0', '42_xm0'] + ['11_xm0', '22_xm0', '33_xm0', '44_xm0']\
#        +['13_xel', '23_xel', '43_xel', '31_xel', '32_xel', '34_xel'] + ['12_xel', '14_xel', '21_xel', '41_xel', '24_xel', '42_xel'] + ['11_xel', '22_xel', '33_xel', '44_xel']
# dtrain = [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0] \
#        + [1, 1, 0, 1, 1, 0] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1]
# dshow = [1, 0, 0, 1, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 1] \
#       + [1, 0, 0, 1, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 1]
# dpercent = [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1] \
#          + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1]

# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dtrain = ' '.join(map(str, dtrain))
# dshow = ' '.join(map(str, dshow))

# L = 4
# suffix = f'{L}L1H'

# # Normal training
# os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} /bin/python -m main -data_size {data_size} -seed 1 -func {target} -lr {lr} -m {model}\
#                 -scheduler {scheduler} -ne 100 -nl {L} -nh 1 -bs {batch_size} -dir_suffix {dir_suffix} \
#                 -dmode {dmode} -dp {dp} -dn {dn} -dtrain {dtrain} -dshow {dshow} -suffix {suffix}\
#                 -ple 1 -pae 10 -plae 10 -sme 500\
#                 --optim_T_max 4000 --optim_eta_min 1e-5 --optim_multiplier 5 --optim_total_epoch 400')


'''
=====================================================================================================================
                                                    Chain-of-Thought Task
=====================================================================================================================
'''
# # GPT Chain Search
# seed_list = [1]
# target = 'chain_search'
# dir_suffix = 'Chain_of_Thought'
# lr = 2e-5
# gpu_id = 1
# batch_size = 100
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# model = 'GPT'
# data_size = 10000

# dname = ['train', 'test']
# dmode = ['train', 'test']
# dtrain = [0, 1]
# dshow = [1, 1]
# dpercent = [9, 1]

# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dtrain = ' '.join(map(str, dtrain))
# dshow = ' '.join(map(str, dshow))

# L, H = 3, 1
# suffix = f'{L}L{H}H'

# # Normal training
# os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} /bin/python -m main -data_size {data_size} -seed 1 -func {target} -lr {lr} -m {model}\
#                 -scheduler {scheduler} -ne 4000 -nl {L} -nh {H} -bs {batch_size} -dir_suffix {dir_suffix} \
#                 -dmode {dmode} -dp {dp} -dn {dn} -dtrain {dtrain} -dshow {dshow} -suffix {suffix}\
#                 -ple 1 -pae 10 -plae 10 -sme 50 -sl 13\
#                 --optim_T_max 4000 --optim_eta_min 1e-5 --optim_multiplier 5 --optim_total_epoch 400')


'''
=====================================================================================================================
                                                    Initial Condensation
=====================================================================================================================
'''
# GPT 3x to x
# target = '3x_to_x'
# dir_suffix = '3x_to_x_task_condense'
# lr = 2e-5
# gpu_id = 0
# batch_size = 100
# scheduler = 'StepLR'
# model = 'GPT_condense'
# data_size = 10000 # Total amount of data for each type

# # xm0 indicates x mod seq_len = 0 is the test set, xel indicates x else, i.e., the training set
# dname = ['train', 'test']
# dmode = ['train', 'test']
# dtrain = [1, 0]
# dshow = [1, 1]
# dpercent = [9, 1] # 90% training set, 10% test set, i.e., training set size is 900, test set size is 100

# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dtrain = ' '.join(map(str, dtrain))
# dshow = ' '.join(map(str, dshow))

# L = 4
# suffix = f'{L}L256H_dk1'

# # Normal training
# os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} python3 -m main -data_size {data_size} -seed 1 -func {target} -lr {lr} -m {model}\
#                 -scheduler {scheduler} -ne 300 -nl {L} -nh 256 -bs {batch_size} -dir_suffix {dir_suffix} \
#                 -dmode {dmode} -dp {dp} -dn {dn} -dtrain {dtrain} -dshow {dshow} -suffix {suffix}\
#                 -ple 1 -pae 10 -plae 10 -sme 1 -lds 100 -ldr 0.95 -dm 32 -dk 1 -dv 1 -d_ff 32' )


# '''
# =====================================================================================================================
#                                             Composite Task: Symmetric or Inference Task
# =====================================================================================================================
# '''
# GPT Composite Function
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--std_rate1', type=float, default=0.8)
# parser.add_argument('--std_rate2', type=float, default=0.5)
parser.add_argument('--optim_multiplier', type=float, default=3)
parser.add_argument('--gpu_id', type=int, default=4)
parser.add_argument('--train_method', type=str, default='train_last_token')
parser.add_argument('--nh', type=int, default=10)
parser.add_argument('--nl', type=int, default=10)
parser.add_argument('--embedding_std', type=float, default=0.5)
parser.add_argument('--qk_std', type=float, default=0.5)
parser.add_argument('--vo_std', type=float, default=0.5)
parser.add_argument('--mlp_std', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

# std_rate = args.std_rate
optim_multiplier = args.optim_multiplier
# std_rate1, std_rate2 = args.std_rate1, args.std_rate2
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
target = 'composition'
# std_rate=0.6
# beta2=0.999
# eps=1e-8
# weight_decay=1e-2
# optim_multiplier=10
train_method = args.train_method
# dir_suffix = f'diff_lr_5e-4_composition_task_34_unseen_43_unseen_diff_ini_{std_rate}_optim_multiplier_{optim_multiplier}'
lr = 1e-5
gpu_id = args.gpu_id
batch_size = 2048
scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# scheduler = 'StepLR'
model = 'GPT2_init_for_diff_part_prenorm'
# model='GPT'
data_size = 200000

# xm0 indicates x mod (seq-1) = 0 is the test set, xel indicates x else, i.e., the training set
dname = ['13_xm0', '23_xm0', '43_xm0', '31_xm0', '32_xm0', '34_xm0'] + ['12_xm0', '14_xm0', '21_xm0', '41_xm0', '24_xm0', '42_xm0'] + ['11_xm0', '22_xm0', '33_xm0', '44_xm0']\
       +['13_xel', '23_xel', '43_xel', '31_xel', '32_xel', '34_xel'] + ['12_xel', '14_xel', '21_xel', '41_xel', '24_xel', '42_xel'] + ['11_xel', '22_xel', '33_xel', '44_xel']
dmode = ['13_xm0', '23_xm0', '43_xm0', '31_xm0', '32_xm0', '34_xm0'] + ['12_xm0', '14_xm0', '21_xm0', '41_xm0', '24_xm0', '42_xm0'] + ['11_xm0', '22_xm0', '33_xm0', '44_xm0']\
       +['13_xel', '23_xel', '43_xel', '31_xel', '32_xel', '34_xel'] + ['12_xel', '14_xel', '21_xel', '41_xel', '24_xel', '42_xel'] + ['11_xel', '22_xel', '33_xel', '44_xel']

dtrain = [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0] \
       + [1, 1, 0, 1, 1, 0] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1]

### The dataset has been modified!!!!!!!!!!

dshow = [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0] \
      + [0, 0, 1, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0]
dpercent = [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1] \
         + [9, 9, 9, 9, 9, 9] + [9, 9, 9, 9, 9, 9] + [9, 9, 9, 9]

dn = ' '.join(map(str, dname))
dp = ' '.join(map(str, dpercent))
dmode = ' '.join(map(str, dmode))
dtrain = ' '.join(map(str, dtrain))
dshow = ' '.join(map(str, dshow))


L, H = args.nl, args.nh

proj_name = f'data_200k_diff_init_part_prenorm_new_init_{train_method}_{target}_{model}_nh_{H}_nl_{L}_without_34'

# proj_name='last_token_nl_2_nh_1_data_90w_normal_init_wo_43_34'

# proj_name='test'

# for seed in [1,2,3,4]:
seed = args.seed
for wd in [0.0, 0.01, 0.1, 1.0]:
# for std_rate in [std_rate1]:
       # for L in [2]:

       dir_suffix = f'embedding_std_{args.embedding_std}_qk_std_{args.qk_std}_vo_std_{args.vo_std}_mlp_std_{args.mlp_std}'
       suffix = f'seed{seed}_wd_{wd}'
       # Normal training
       os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} python3 -m main -data_size {data_size} -seed {seed} -func {target} -lr {lr} -m {model}\
                     -scheduler {scheduler} -ne 1050 -nl {L} -nh {H} -bs {batch_size} -dir_suffix {dir_suffix} -pname {proj_name} -dk 64 -dv 64 -d_ff 1280 -dm 640 \
                     -dmode {dmode} -dp {dp} -dn {dn} -dtrain {dtrain} -dshow {dshow} -suffix {suffix}\
                     -ple 1 -pae 30 -plae 30 -sme 200 -wd {wd} -embedding_std {args.embedding_std} -qk_std {args.qk_std} -vo_std {args.vo_std} -mlp_std {args.mlp_std}\
                     --optim_T_max 1000 --optim_eta_min 1e-5 --optim_multiplier {optim_multiplier} --optim_total_epoch 50')
# '''
# ======================================================================================================================
#                             Composite Task: Symmetric or Inference Task, Analyzing Data Complexity Impact on Training Speed
# ======================================================================================================================
# '''
# # GPT Composite Function
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--std_rate', type=float, default=0.3)
# parser.add_argument('--optim_multiplier', type=float, default=10)
# parser.add_argument('--gpu_id', type=int, default=0)
# parser.add_argument('--random_data_num', type=int, default=0)
# parser.add_argument('--L', type=int, default=2)
# args = parser.parse_args()

# std_rate = args.std_rate
# optim_multiplier = args.optim_multiplier
# random_data_num = args.random_data_num
# # import os
# # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# target = 'composition_random'
# # std_rate=0.6
# # beta2=0.999
# # eps=1e-8
# # weight_decay=1e-2
# # optim_multiplier=10
# dir_suffix = f'diff_lr_5e-4_composition_task_34_unseen_43_unseen_random_num_{random_data_num}_diff_ini_{std_rate}_optim_multiplier_{optim_multiplier}'
# lr = 1e-5
# gpu_id = args.gpu_id
# batch_size = 300
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# # scheduler = 'StepLR'
# model = 'GPT_normal_init'
# # model='GPT'
# data_size = 100000

# # xm0 indicates x mod (seq-1) = 0 is the test set, xel indicates x else, i.e., the training set
# dname = ['13_xm0', '23_xm0', '43_xm0', '31_xm0', '32_xm0', '34_xm0'] + ['12_xm0', '14_xm0', '21_xm0', '41_xm0', '24_xm0', '42_xm0'] + ['11_xm0', '22_xm0', '33_xm0', '44_xm0']\
#        +['13_xel', '23_xel', '43_xel', '31_xel', '32_xel', '34_xel'] + ['12_xel', '14_xel', '21_xel', '41_xel', '24_xel', '42_xel'] + ['11_xel', '22_xel', '33_xel', '44_xel']
# dmode = ['13_xm0', '23_xm0', '43_xm0', '31_xm0', '32_xm0', '34_xm0'] + ['12_xm0', '14_xm0', '21_xm0', '41_xm0', '24_xm0', '42_xm0'] + ['11_xm0', '22_xm0', '33_xm0', '44_xm0']\
#        +['13_xel', '23_xel', '43_xel', '31_xel', '32_xel', '34_xel'] + ['12_xel', '14_xel', '21_xel', '41_xel', '24_xel', '42_xel'] + ['11_xel', '22_xel', '33_xel', '44_xel']

# dtrain = [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0] \
#        + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1]

# ### The dataset has been modified!!!!!!!!!!

# dshow = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 1] \
#       + [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 1]
# dpercent = [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1] \
#          + [9, 9, 9, 9, 9, 9] + [9, 9, 9, 9, 9, 9] + [9, 9, 9, 9]

# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dtrain = ' '.join(map(str, dtrain))
# dshow = ' '.join(map(str, dshow))

# L = args.L

# proj_name = 'data_complexity_epoch_207_data_900k_warmup_normal_init_wo_43_34_diff_ini'

# # proj_name='test'

# for seed in [2,3,4,5,6,7,8,9]:
#        # if seed==2:
#        #        L_list=[6,7]
#        # else:
#        # L_list=[2,3,4,5,6,7]
#        # L_list=[2,4,6]
#        # L_list=[2,3,4]
#        # for L in L_list:
#        # for L in [2,4,6, 8]:
#        # for L in [3,5,7]:
#        # 
#        # for L in [2,7]:
#        # for L in [3,6]:
#        # for L in [4,5]:
#        # L = 3
#        suffix = f'{L}L1H_seed{seed}'
#        # Normal training
#        os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} python3 -m main -data_size {data_size} -seed {seed} -func {target} -lr {lr} -m {model}\
#                      -scheduler {scheduler} -ne 21 -nl {L} -nh 1 -bs {batch_size} -dir_suffix {dir_suffix} -pname {proj_name} -dk 200 -dv 200\
#                      -dmode {dmode} -dp {dp} -dn {dn} -dtrain {dtrain} -dshow {dshow} -suffix {suffix}\
#                      -ple 1 -pae 3 -plae 3 -sme 20 -sr {std_rate} -rdm {random_data_num} \
#                      --optim_T_max 200 --optim_eta_min 1e-5 --optim_multiplier {optim_multiplier} --optim_total_epoch 10')


# '''
# ======================================================================================================================
#                                 Extremely Small Initialization: Research Target
# ======================================================================================================================
# '''
# # GPT Composite Function
# # import argparse

# # parser = argparse.ArgumentParser()
# # parser.add_argument('--std_rate', type=float, default=0.6)
# # parser.add_argument('--optim_multiplier', type=float, default=10)
# # parser.add_argument('--gpu_id', type=int, default=0)
# # args = parser.parse_args()

# std_rate = 3
# optim_multiplier = 10
# # import os
# # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# target = 'composition'
# # std_rate=0.6
# # beta2=0.999
# # eps=1e-8
# # weight_decay=1e-2
# # optim_multiplier=10
# dir_suffix = f'no_34_diff_lr_5e-4_composition_task_34_unseen_43_unseen_diff_ini_{std_rate}_optim_multiplier_{optim_multiplier}'
# lr = 1e-5
# gpu_id = 1
# batch_size = 2048
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# # scheduler = 'StepLR'
# model = 'GPT_normal_init'
# # model='GPT'
# data_size = 900000

# # xm0 indicates x mod (seq-1) = 0 is the test set, xel indicates x else, i.e., the training set
# dname = ['13_xm0', '23_xm0', '43_xm0', '31_xm0', '32_xm0', '34_xm0'] + ['12_xm0', '14_xm0', '21_xm0', '41_xm0', '24_xm0', '42_xm0'] + ['11_xm0', '22_xm0', '33_xm0', '44_xm0']\
#        +['13_xel', '23_xel', '43_xel', '31_xel', '32_xel', '34_xel'] + ['12_xel', '14_xel', '21_xel', '41_xel', '24_xel', '42_xel'] + ['11_xel', '22_xel', '33_xel', '44_xel']
# dmode = ['13_xm0', '23_xm0', '43_xm0', '31_xm0', '32_xm0', '34_xm0'] + ['12_xm0', '14_xm0', '21_xm0', '41_xm0', '24_xm0', '42_xm0'] + ['11_xm0', '22_xm0', '33_xm0', '44_xm0']\
#        +['13_xel', '23_xel', '43_xel', '31_xel', '32_xel', '34_xel'] + ['12_xel', '14_xel', '21_xel', '41_xel', '24_xel', '42_xel'] + ['11_xel', '22_xel', '33_xel', '44_xel']

# dtrain = [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0] \
#        + [1, 1, 0, 1, 1, 0] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1]

# ### The dataset has been modified!!!!!!!!!!

# dshow = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 1] \
#       + [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 1]
# dpercent = [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1] \
#          + [9, 9, 9, 9, 9, 9] + [9, 9, 9, 9, 9, 9] + [9, 9, 9, 9]

# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dtrain = ' '.join(map(str, dtrain))
# dshow = ' '.join(map(str, dshow))


# proj_name = 'phase_diagram_for_no34_43_4head_epoch_207_data_900k_warmup_normal_init_wo_43_unseen_diff_ini'

# # proj_name='test'


# for L in [2]:
# # for L in [2,4,6, 8]:
# # for L in [3,5,7]:
# # 
# # for L in [2,7]:
# # for L in [3,6]:
# # for L in [4,5]:
# # L = 3
#        suffix = f'{L}L1H_seed1'
#        # Normal training
#        os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} python3 -m main -data_size {data_size} -seed 1 -func {target} -lr {lr} -m {model}\
#                      -scheduler {scheduler} -ne 210 -nl {L} -nh 1 -bs {batch_size} -dir_suffix {dir_suffix} -pname {proj_name} -dk 200 -dv 200\
#                      -dmode {dmode} -dp {dp} -dn {dn} -dtrain {dtrain} -dshow {dshow} -suffix {suffix}\
#                      -ple 1 -pae 3 -plae 3 -sme 20 -sr {std_rate} \
#                      --optim_T_max 200 --optim_eta_min 1e-5 --optim_multiplier {optim_multiplier} --optim_total_epoch 10')


# '''
# ======================================================================================================================
#                                 Composite Task: Symmetric or Inference Task, Saving Every Epoch to Observe Condensation
# ======================================================================================================================
# '''
# target = 'composition'
# std_rate = 0.8
# beta2 = 0.999
# eps = 1e-8
# weight_decay = 1e-2
# dir_suffix = f'diff_lr_1e-5_composition_task_34_unseen_43_unseen_diff_ini_{std_rate}_test_eps_{eps}_wd_{weight_decay}_beta2_{beta2}_for_test_diff_epoch_condense'
# lr = 1e-5
# gpu_id = 4
# batch_size = 2048
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# # scheduler = 'StepLR'
# model = 'GPT_normal_init'
# # model='GPT'
# data_size = 900000

# # xm0 indicates x mod (seq-1) = 0 is the test set, xel indicates x else, i.e., the training set
# dname = ['13_xm0', '23_xm0', '43_xm0', '31_xm0', '32_xm0', '34_xm0'] + ['12_xm0', '14_xm0', '21_xm0', '41_xm0', '24_xm0', '42_xm0'] + ['11_xm0', '22_xm0', '33_xm0', '44_xm0']\
#        +['13_xel', '23_xel', '43_xel', '31_xel', '32_xel', '34_xel'] + ['12_xel', '14_xel', '21_xel', '41_xel', '24_xel', '42_xel'] + ['11_xel', '22_xel', '33_xel', '44_xel']
# dmode = ['13_xm0', '23_xm0', '43_xm0', '31_xm0', '32_xm0', '34_xm0'] + ['12_xm0', '14_xm0', '21_xm0', '41_xm0', '24_xm0', '42_xm0'] + ['11_xm0', '22_xm0', '33_xm0', '44_xm0']\
#        +['13_xel', '23_xel', '43_xel', '31_xel', '32_xel', '34_xel'] + ['12_xel', '14_xel', '21_xel', '41_xel', '24_xel', '42_xel'] + ['11_xel', '22_xel', '33_xel', '44_xel']

# dtrain = [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0] \
#        + [1, 1, 0, 1, 1, 1] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1]

# ### The dataset has been modified!!!!!!!!!!

# dshow = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 1] \
#       + [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 1]
# dpercent = [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1] \
#          + [9, 9, 9, 9, 9, 9] + [9, 9, 9, 9, 9, 9] + [9, 9, 9, 9]

# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dtrain = ' '.join(map(str, dtrain))
# dshow = ' '.join(map(str, dshow))


# proj_name = 'refine_adam_epoch_207_data_900k_warmup_normal_init_wo_43_unseen_diff_ini'

# # proj_name='test'


# for L in [2]:
# # for L in [2,4,6, 8]:
# # for L in [3,5,7]:
# # 
# # for L in [2,7]:
# # for L in [3,6]:
# # for L in [4,5]:
# # L = 3
#        suffix = f'{L}L1H_seed1'
#        # Normal training
#        os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} python3 -m main -data_size {data_size} -seed 1 -func {target} -lr {lr} -m {model}\
#                      -scheduler {scheduler} -ne 20 -nl {L} -nh 1 -bs {batch_size} -dir_suffix {dir_suffix} -pname {proj_name} -dk 200 -dv 200\
#                      -dmode {dmode} -dp {dp} -dn {dn} -dtrain {dtrain} -dshow {dshow} -suffix {suffix}\
#                      -ple 1 -pae 1 -plae 1 -sme 1 -sr {std_rate} -beta2 {beta2} -eps {eps} -wd {weight_decay} \
#                      --optim_T_max 200 --optim_eta_min 1e-5 --optim_multiplier 20 --optim_total_epoch 10')


'''
=====================================================================================================================
                                                    Composite Task: Symmetric or Inference Task
=====================================================================================================================
'''
# # GPT Composite Function
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--std_rate', type=float, default=0.5)
# parser.add_argument('--optim_multiplier', type=float, default=20)
# parser.add_argument('--gpu_id', type=int, default=5)

# args = parser.parse_args()

# std_rate = args.std_rate
# optim_multiplier = args.optim_multiplier
# # import os
# # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# target = 'composition'
# # std_rate=0.6
# # beta2=0.999
# # eps=1e-8
# # weight_decay=1e-2
# # optim_multiplier=10
# dir_suffix = f'diff_lr_5e-4_composition_task_34_unseen_43_unseen_diff_ini_{std_rate}_optim_multiplier_{optim_multiplier}'
# lr = 1e-5
# gpu_id = args.gpu_id
# batch_size = 2048
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# # scheduler = 'StepLR'
# model = 'GPT_normal_init'
# # model='GPT'
# data_size = 900000

# # Initialize lists
# dname, dmode, dtrain, dshow, dpercent = [], [], [], [], []

# # Define prefixes and suffixes
# # prefixes = ['13', '23', '43', '31', '32', '34', '12', '14', '21', '41', '24', '42', '11', '22', '33', '44']
# prefixes = [f"{i}{j}" for i in range(1, 5) for j in range(1, 5)]
# suffixes = ['xm0', 'xel']

# # Generate dname and dmode
# for suffix in suffixes:
#     for prefix in prefixes:
#         dname.append(f"{prefix}_{suffix}")
# dmode = dname.copy()


# # Initialize train_values and show_values
# dtrain = [0 if 'xm0' in name else 1 for name in dname]
# dshow = [0] * len(dname)

# # Specific lists that need to be updated
# specific_list = ['43_xel', '34_xel']  # Example, replace as needed

# # Update train_values and show_values
# for name in specific_list:
#     index = dname.index(name)
#     # Assume we set train_values corresponding position to 0, show_values to 1
#     # Adjust these values as needed
#     dtrain[index] = 0  # Example value

# specific_list2 = ['43_xel']  # Example, replace as needed

# # Update train_values and show_values
# for name in specific_list2:
#     index = dname.index(name)
#     # Assume we set show_values corresponding position to 1
#     # Adjust these values as needed
# #     dtrain[index] = 0  # Example value
#     dshow[index] = 1  # Example value


# # Define values for dtrain, dshow, dpercent
# # train_values = [0]*6 + [0]*6 + [0]*4 + [1]*6 + [1]*6 + [1]*4
# # show_values = [0, 0, 1, 0, 0, 1] + [0]*6 + [0, 0, 0, 1] + [0, 0, 1, 0, 0, 1] + [0]*6 + [0, 0, 0, 1]
# dpercent = [1]*int(len(dname)/2) + [9]*int(len(dname)/2)

# # # xm0 indicates x mod (seq-1) = 0 is the test set, xel indicates x else, i.e., the training set
# # dname = ['13_xm0', '23_xm0', '43_xm0', '31_xm0', '32_xm0', '34_xm0'] + ['12_xm0', '14_xm0', '21_xm0', '41_xm0', '24_xm0', '42_xm0'] + ['11_xm0', '22_xm0', '33_xm0', '44_xm0']\
# #        +['13_xel', '23_xel', '43_xel', '31_xel', '32_xel', '34_xel'] + ['12_xel', '14_xel', '21_xel', '41_xel', '24_xel', '42_xel'] + ['11_xel', '22_xel', '33_xel', '44_xel']
# # # dmode = ['13_xm0', '23_xm0', '43_xm0', '31_xm0', '32_xm0', '34_xm0'] + ['12_xm0', '14_xm0', '21_xm0', '41_xm0', '24_xm0', '42_xm0'] + ['11_xm0', '22_xm0', '33_xm0', '44_xm0']\
# # #        +['13_xel', '23_xel', '43_xel', '31_xel', '32_xel', '34_xel'] + ['12_xel', '14_xel', '21_xel', '41_xel', '24_xel', '42_xel'] + ['11_xel', '22_xel', '33_xel', '44_xel']

# # dmode = dname
# # dtrain = [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0] \
# #        + [1, 1, 0, 1, 1, 1] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1]

# # ### The dataset has been modified!!!!!!!!!!

# # dshow = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 1] \
# #       + [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 1]
# # dpercent = [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1] \
# #          + [9, 9, 9, 9, 9, 9] + [9, 9, 9, 9, 9, 9] + [9, 9, 9, 9]

# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dtrain = ' '.join(map(str, dtrain))
# dshow = ' '.join(map(str, dshow))


# proj_name = 'phase_diagram_for_no34_43_4head_epoch_207_data_900k_warmup_normal_init_wo_43_unseen_diff_ini'

# # proj_name='test'

# for seed in [1, 2, 3]:
#        for L in [2, 3, 4, 5, 6, 7]:
#        # for L in [2,4,6, 8]:
#        # for L in [3,5,7]:
#        # 
#        # for L in [2,7]:
#        # for L in [3,6]:
#        # for L in [4,5]:
#        # L = 3
#               suffix = f'{L}L1H_seed{seed}'
#               # Normal training
#               os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} python3 -m main -data_size {data_size} -seed {seed} -func {target} -lr {lr} -m {model}\
#                             -scheduler {scheduler} -ne 210 -nl {L} -nh 4 -bs {batch_size} -dir_suffix {dir_suffix} -pname {proj_name} -dk 64 -dv 64\
#                             -dmode {dmode} -dp {dp} -dn {dn} -dtrain {dtrain} -dshow {dshow} -suffix {suffix}\
#                             -ple 1 -pae 3 -plae 3 -sme 20 -sr {std_rate} \
#                             --optim_T_max 200 --optim_eta_min 1e-5 --optim_multiplier {optim_multiplier} --optim_total_epoch 10')


# '''
# ======================================================================================================================
#                                            Very Small Initialization: Finding a Research Subject
# ======================================================================================================================
# '''
# # GPT Composite Function
# # import argparse

# # parser = argparse.ArgumentParser()
# # parser.add_argument('--std_rate', type=float, default=0.6)
# # parser.add_argument('--optim_multiplier', type=float, default=10)
# # parser.add_argument('--gpu_id', type=int, default=0)
# # args = parser.parse_args()

# std_rate = 3
# optim_multiplier = 10
# # import os
# # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# target = 'composition'
# # std_rate=0.6
# # beta2=0.999
# # eps=1e-8
# # weight_decay=1e-2
# # optim_multiplier=10
# dir_suffix = f'no_34_diff_lr_5e-4_composition_task_34_unseen_43_unseen_diff_ini_{std_rate}_optim_multiplier_{optim_multiplier}'
# lr = 1e-5
# gpu_id = 1
# batch_size = 2048
# scheduler = 'GradualWarmupScheduler_CosineAnnealingLR'
# # scheduler = 'StepLR'
# model = 'GPT_normal_init'
# # model='GPT'
# data_size = 900000

# # xm0 indicates x mod (seq-1) = 0 is the test set, xel indicates x else, i.e., the training set
# dname = ['13_xm0', '23_xm0', '43_xm0', '31_xm0', '32_xm0', '34_xm0'] + ['12_xm0', '14_xm0', '21_xm0', '41_xm0', '24_xm0', '42_xm0'] + ['11_xm0', '22_xm0', '33_xm0', '44_xm0']\
#        +['13_xel', '23_xel', '43_xel', '31_xel', '32_xel', '34_xel'] + ['12_xel', '14_xel', '21_xel', '41_xel', '24_xel', '42_xel'] + ['11_xel', '22_xel', '33_xel', '44_xel']
# dmode = ['13_xm0', '23_xm0', '43_xm0', '31_xm0', '32_xm0', '34_xm0'] + ['12_xm0', '14_xm0', '21_xm0', '41_xm0', '24_xm0', '42_xm0'] + ['11_xm0', '22_xm0', '33_xm0', '44_xm0']\
#        +['13_xel', '23_xel', '43_xel', '31_xel', '32_xel', '34_xel'] + ['12_xel', '14_xel', '21_xel', '41_xel', '24_xel', '42_xel'] + ['11_xel', '22_xel', '33_xel', '44_xel']

# dtrain = [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 0] \
#        + [1, 1, 0, 1, 1, 0] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1]

# ### The dataset has been modified!!!!!!!!!!

# dshow = [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 1] \
#       + [0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0] + [0, 0, 0, 1]
# dpercent = [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1, 1, 1] + [1, 1, 1, 1] \
#          + [9, 9, 9, 9, 9, 9] + [9, 9, 9, 9, 9, 9] + [9, 9, 9, 9]

# dn = ' '.join(map(str, dname))
# dp = ' '.join(map(str, dpercent))
# dmode = ' '.join(map(str, dmode))
# dtrain = ' '.join(map(str, dtrain))
# dshow = ' '.join(map(str, dshow))


# proj_name = 'phase_diagram_for_no34_43_4head_epoch_207_data_900k_warmup_normal_init_wo_43_unseen_diff_ini'

# # proj_name='test'


# for seed in [1, 2, 3]:
#        for L in [2, 3, 4, 5, 6, 7]:
#        # for L in [2,4,6, 8]:
#        # for L in [3,5,7]:
#        # 
#        # for L in [2,7]:
#        # for L in [3,6]:
#        # for L in [4,5]:
#        # L = 3
#               suffix = f'{L}L1H_seed1'
#               # Normal training
#               os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} python3 -m main -data_size {data_size} -seed 1 -func {target} -lr {lr} -m {model}\
#                             -scheduler {scheduler} -ne 20 -nl {L} -nh 1 -bs {batch_size} -dir_suffix {dir_suffix} -pname {proj_name} -dk 200 -dv 200\
#                             -dmode {dmode} -dp {dp} -dn {dn} -dtrain {dtrain} -dshow {dshow} -suffix {suffix}\
#                             -ple 1 -pae 1 -plae 1 -sme 1 -sr {std_rate} -beta2 {beta2} -eps {eps} -wd {weight_decay} \
#                             --optim_T_max 200 --optim_eta_min 1e-5 --optim_multiplier 20 --optim_total_epoch 10')
