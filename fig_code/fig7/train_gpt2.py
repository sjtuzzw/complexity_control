#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""


import string, random, os
import pickle
import argparse


my_parser = argparse.ArgumentParser(description="Pytorch distributed")
my_parser.add_argument('-datasize', '--datasize', type = str, default = '20')
my_parser.add_argument('-epoch', '--epoch', type = int, default = 20)

my_args, remaining = my_parser.parse_known_args()

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names




def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.


# wvwo_id is the multiplier in front of WVWO identity scaling
# wkwq_id is the multiplier in front of WKWQ identity scaling
# If either is 0, then the corresponding part is turned off
identity_training_args = {'wvwo_id' : 0, 'wkwq_id' : 1}


wvwo_id = identity_training_args['wvwo_id']
wkwq_id = identity_training_args['wkwq_id']

wvwo_str = '' if wvwo_id == 0 else f'wvwo{wvwo_id}_'
wkwq_str = '' if wkwq_id == 0 else f'wkwq{wkwq_id}_'

mask_diag_wkwq = True
mask_diag_str = '' if not mask_diag_wkwq else 'maskwkwqiddiag_'
# import argparse

# ini_parser = argparse.ArgumentParser()

# ini_parser.add_argument('--lr', type=float, default=5e-5)
# args = ini_parser.parse_args()
lr = 5e-5
task_type = f'gpt2_adam{lr}_{wvwo_str}{wkwq_str}{mask_diag_str}'


curr_task_name = task_type + get_random_string(4)
while os.path.exists(f'./logs/{curr_task_name}'):
    random.seed(time.time())
    curr_task_name = task_type + get_random_string(4)
print(curr_task_name)

arg_list = ['--output_dir',f'./save_training_ini_rate_0.3_wd_0.0_last_token_{my_args.datasize}',
            # '--resume_from_checkpoint', '/home/zhangzhongwang/data/train_gpt2_on_SCAN/train_gpt2/save_training_ini_rate_default_0.01_wd_0.01_point042w_refined_token/checkpoint-2000',
            '--save_total_limit', '1',
           '--model_type', 'gpt2',
           '--tokenizer_name', 'gpt2',
            
        #    '--dataset_name', 'wikitext-2-v1',
        #    '--dataset_name', f'scan_{my_args.datasize}w_refined_token', 
            '--dataset_name', f'cogs_{my_args.datasize}', 
        #    '--dataset_config_name', 'wikitext-2-v1',
            
            # '--dataset_name', 'oscar',
            # '--dataset_config_name', 'unshuffled_deduplicated_als', # 459,001 words, 2.8M (Alemmanisch)
            # '--dataset_config_name', 'unshuffled_deduplicated_bh', # 2,875 words, 34K (Bihari)
            
            # '--auto_find_batch_size',
            '--per_device_train_batch_size', '20',
            '--per_device_eval_batch_size', '20',
            
            # '--max_steps', '1',
            # '--optim', 'sgd',
            '--weight_decay', '0.0',
            '--optim', 'adamw_torch_fused',
           '--learning_rate', f'{lr}',
           '--do_train',
           '--do_eval',
           '--no_skip_memory_metrics',
           '--overwrite_output_dir',
           '--save_safetensors',
           '--num_train_epochs', f'{my_args.epoch}',
           '--save_steps', '2000',
           '--logging_steps', '1',
           '--logging_dir', f'./logs/{curr_task_name}',
            '--evaluation_strategy', 'steps',
            '--eval_steps', '100',
            '--lr_scheduler_type', 'constant',
            # '--fp16',
            # '--torch_compile', # Not yet supported for Python 3.11 for some reason....
           ]

pickle.dump({'args' : arg_list, 'identity_args' : identity_training_args}, open(f'./train_configs/{curr_task_name}.pkl', 'wb'))


# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import time

import datasets
import evaluate
import torch
from datasets import load_dataset
import nvidia_smi
import numpy as np
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    TrainerCallback,
)

from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# from transformers import GPT2Config, GPT2LMHeadModel
# from gpt2_with_identity import GPT2Config, GPT2LMHeadModel_WithIdentity
# from gpt2_origin import GPT2Config, GPT2LMHeadModel
from gpt2_origin_init_last_token import GPT2Config, GPT2LMHeadModel
# from gpt2_origin_last_token import GPT2Config, GPT2LMHeadModel
# from gpt2_origin_init import GPT2Config, GPT2LMHeadModel
# from gpt2_origin_noised import GPT2Config, GPT2LMHeadModel

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.32.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
print(arg_list)

model_args, data_args, training_args = parser.parse_args_into_dataclasses(arg_list)


if model_args.use_auth_token is not None:
    warnings.warn("The `use_auth_token` argument is deprecated and will be removed in v4.34.", FutureWarning)
    if model_args.token is not None:
        raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
    model_args.token = model_args.use_auth_token

# # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
# # information sent is the one passed as arguments along with your Python/PyTorch versions.
# send_example_telemetry("run_clm", model_args, data_args)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if training_args.should_log:
    # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    transformers.utils.logging.set_verbosity_info()

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")
if not os.path.isdir(training_args.output_dir):
    os.mkdir(training_args.output_dir)
# Detecting last checkpoint.
last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

# Set seed before initializing model.
set_seed(training_args.seed)

# Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
# (the dataset will be downloaded automatically from the datasets Hub).
#
# For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
# 'text' is found. You can easily tweak this behavior (see below).
#
# In distributed training, the load_dataset function guarantee that only one local process can concurrently
# download the dataset.
if data_args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        data_args.dataset_name,
        # data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        streaming=data_args.streaming,
    )
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            data_args.dataset_name,
            # data_args.dataset_config_name,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            # data_args.dataset_config_name,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
else:
    data_files = {}
    dataset_args = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    extension = (
        data_args.train_file.split(".")[-1]
        if data_args.train_file is not None
        else data_args.validation_file.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        **dataset_args,
    )
    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )

# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
# https://huggingface.co/docs/datasets/loading_datasets.html.

# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.

config_kwargs = {
    "cache_dir": model_args.cache_dir,
    "revision": model_args.model_revision,
    "token": model_args.token,
}
if model_args.config_name:
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
elif model_args.model_name_or_path:
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
else:
    config = CONFIG_MAPPING[model_args.model_type]()
    logger.warning("You are instantiating a new config instance from scratch.")
    if model_args.config_overrides is not None:
        logger.info(f"Overriding config: {model_args.config_overrides}")
        config.update_from_string(model_args.config_overrides)
        logger.info(f"New config: {config}")

config.attn_pdrop=0.0
config.embd_pdrop=0.0
config.resid_pdrop=0.0
config.summary_first_dropout=0.0

tokenizer_kwargs = {
    "cache_dir": model_args.cache_dir,
    "use_fast": model_args.use_fast_tokenizer,
    "revision": model_args.model_revision,
    "token": model_args.token,
}
if model_args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
elif model_args.model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
else:
    raise ValueError(
        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    )

if model_args.model_name_or_path:
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )
else:
    # Original:
    # model = AutoModelForCausalLM.from_config(config)
    
    # New GPT2 model with identity:
    config.wvwo_id_mult = identity_training_args['wvwo_id']
    config.wkwq_id_mult = identity_training_args['wkwq_id']
    config.mask_diag_wkwq = mask_diag_wkwq
    config.eos_id=tokenizer.eos_token_id
    assert(type(config) == GPT2Config)
    # model = GPT2LMHeadModel_WithIdentity(config)
    model = GPT2LMHeadModel(config)  # wzw add
    
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")



model.cuda()


# batch_size = 8
# seqlen = 1024
# rand_input = torch.randint(50000,(batch_size,seqlen)).cuda()



# with torch.no_grad():
#     a = model(rand_input)


# We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
# on a small vocab and want a smaller embedding size, remove this test.
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

# Preprocessing the datasets.
# First we tokenize all the texts.
if training_args.do_train:
    column_names = list(raw_datasets["train"].features)
else:
    column_names = list(raw_datasets["validation"].features)
text_column_name = "text" if "text" in column_names else column_names[0]

# since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")


tokenizer.pad_token = tokenizer.eos_token # wzw加
def tokenize_function(examples):
    with CaptureLogger(tok_logger) as cl:
        # output = tokenizer(examples[text_column_name])
        # wzw改
        output = tokenizer(examples[text_column_name], padding='max_length', truncation=True, max_length=190)
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )
    return output

with training_args.main_process_first(desc="dataset map tokenization"):
    if not data_args.streaming:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    else:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )

if data_args.block_size is None:
    block_size = tokenizer.model_max_length
    if block_size > 1024:
        logger.warning(
            "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
            " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
            " override this default with `--block_size xxx`."
        )
        block_size = 1024
else:
    if data_args.block_size > tokenizer.model_max_length:
        logger.warning(
            f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        )
    block_size = min(data_args.block_size, tokenizer.model_max_length)

print(block_size)

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
# def group_texts(examples):
#     # max_len = 0
#     # for i, example in enumerate(examples['input_ids']):
#     #     if i < 3:
#     #         print(i, example)
#     #     max_len = max(max_len, len(example))
#     # print(max_len, 'max_len')

#     for keys in examples.keys():
#         print('examples', keys, len(examples[keys][0]))


#     # Concatenate all texts.
#     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
#     # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
#     total_length = (total_length // block_size) * block_size
#     # Split by chunks of max_len.
#     result = {
#         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()

#     for keys in result.keys():
#         print('result', keys, len(result[keys][0]))

#     return result

# def group_texts(examples):
#     max_length = 256
    
#     # 对每个句子进行补零
#     padded_examples = []
#     padded_attention_masks = []
#     for sentence in examples['input_ids']:
#         if len(sentence) <= max_length:
#             # 如果句子长度小于等于max_length,则在末尾补0
#             padded_sentence = sentence + [0] * (max_length - len(sentence))
#             padded_examples.append(padded_sentence)
#         else:
#             pass
    
#     for attention_mask in examples['attention_mask']:
#         if len(attention_mask) <= max_length:
#             # 如果句子长度小于等于max_length,则在末尾补0
#             padded_attention_mask = attention_mask + [1] * (max_length - len(attention_mask))
#             padded_attention_masks.append(padded_attention_mask)
#         else:
#             pass
    
#     # 返回补零后的句子列表
#     return {'input_ids': padded_examples, 'labels': padded_examples, 'attention_mask': padded_attention_masks}


def group_texts(examples):
    max_length = 256
    
    # 对每个句子进行补零
    example_list = []
    attention_mask_list = []
    size = len(examples['input_ids'])

    for i in range(size):
        example_list.append(examples['input_ids'][i])
        attention_mask_list.append(examples['attention_mask'][i])

    
    # 返回补零后的句子列表
    return {'input_ids': example_list, 'labels': example_list, 'attention_mask': attention_mask_list}





# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
# for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
# to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

with training_args.main_process_first(desc="grouping texts together"):
    if not data_args.streaming:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    else:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )

# raise ValueError('Not implemented')

if training_args.do_train:
    if "train" not in tokenized_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = lm_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

# print(train_dataset[0])
# print(train_dataset[1])

if training_args.do_eval:
    if "validation" not in tokenized_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = {"train": lm_datasets["validation"], "test": lm_datasets["test"]}
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    # def compute_metrics(eval_preds):
    #     preds, labels = eval_preds

    #     # 预测和标签的形状与经过 argmax(-1) 计算后的形状相同
    #     shift_labels = labels[:, 1:]
    #     shift_logits = preds[:, :-1]

    #     # 使用 numpy 创建结束标志
    #     eos_mask = (shift_labels == 16289)
    #     cumsum_eos_mask = np.cumsum(eos_mask, axis=1)
    #     last_token_index = np.sum(cumsum_eos_mask == 0, axis=1)

    #     batch_indices = np.arange(shift_logits.shape[0])
    #     shift_logits = shift_logits[batch_indices, last_token_index:].reshape(-1)
    #     shift_labels = shift_labels[batch_indices, last_token_index:].reshape(-1)

    #     return metric.compute(predictions=shift_logits, references=shift_labels)
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # 预测和标签的形状与经过 argmax(-1) 计算后的形状相同
        shift_labels = labels[:, 1:]
        shift_logits = preds[:, :-1]

        # 使用 numpy 创建结束标志
        bos_mask = (shift_labels == 16289)
        eos_mask = (shift_labels == config.eos_token_id)
        cumsum_bos_mask = np.cumsum(bos_mask, axis=1)
        cumsum_eos_mask = np.cumsum(eos_mask, axis=1)
        first_token_index = np.sum(cumsum_bos_mask == 0, axis=1)
        last_token_index = np.sum(cumsum_eos_mask == 0, axis=1)

        correct_samples = 0
        total_samples = shift_logits.shape[0]

        for i in range(total_samples):
            sample_preds = shift_logits[i, first_token_index[i]+2:last_token_index[i]]
            sample_labels = shift_labels[i, first_token_index[i]+2:last_token_index[i]]

            # 检查是否所有token都匹配
            if np.array_equal(sample_preds, sample_labels):
                correct_samples += 1
            # else:
            #     print(sample_preds, sample_labels)

        accuracy = correct_samples / total_samples

        return {"accuracy": accuracy}

tb_writer = SummaryWriter(log_dir=training_args.logging_dir + '_mem')
class MemoryLogCallback(TensorBoardCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        nvidia_smi.nvmlInit()
        try:
            deviceCount = nvidia_smi.nvmlDeviceGetCount()
            for i in range(deviceCount):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                dev_name = nvidia_smi.nvmlDeviceGetName(handle)
                self.tb_writer.add_scalar(f"dev{i}/used_mem_bytes", info.used, state.global_step)
                self.tb_writer.add_scalar(f"dev{i}/free_mem_bytes", info.free, state.global_step)
                # print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, , 100*info.free/info.total, info.total, info.free, info.used))
                # print(dir(info))
        except:
            pass
        nvidia_smi.nvmlShutdown()

ml_callback = MemoryLogCallback(tb_writer)




all_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)

bias_parameters = [name for name in all_parameters if "bias" in name]
decay_parameters = [name for name in all_parameters if 'bias' not in name]



optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
        ],
        "weight_decay": training_args.weight_decay,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if (n in bias_parameters and p.requires_grad)
        ],
        "weight_decay": 0.0,
    },
]


optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)

optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)




# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    # data_collator=default_data_collator,
    compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
    if training_args.do_eval and not is_torch_tpu_available()
    else None,
    callbacks=[ml_callback],
    optimizers=(optimizer,None)
)


# Training
if training_args.do_train:
    checkpoint = None
    if training_args.resume_from_checkpoint is None:
        print('Overwriting Model0')
        trainer.save_model(output_dir=f'{training_args.output_dir}/checkpoint-0')
    else:
        print('Not overwriting model0')
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    


    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


# Evaluation
if training_args.do_eval:
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate()

    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
if data_args.dataset_name is not None:
    kwargs["dataset_tags"] = data_args.dataset_name
    if data_args.dataset_config_name is not None:
        kwargs["dataset_args"] = data_args.dataset_config_name
        kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
    else:
        kwargs["dataset"] = data_args.dataset_name

if training_args.push_to_hub:
    trainer.push_to_hub(**kwargs)
else:
    trainer.create_model_card(**kwargs)
