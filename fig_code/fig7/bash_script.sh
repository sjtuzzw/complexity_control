#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch train_gpt2.py --datasize 1w --epoch 5000 > output_0.3_wd_0.0_no_droP_last_token_1w.log 2>&1;
