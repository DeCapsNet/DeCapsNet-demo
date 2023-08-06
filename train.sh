#!/bin/bash

ran=$((1000 + RANDOM % 9000))
CUDA_VISIBLE_DEVICES=2 python -u run.py \
        --seed $ran \
        --input_dir "../../data/depression/RSDD/processed/phq9_temp2_top64" \
        --save_path "../../data/depression/RSDD/result/model_$ran.pth" \
        --save_log_path "../../data/depression/RSDD/result/result.txt" \
        --model_type "bert-base-uncased"\
        --train_batch_size 16 \
        --test_batch_size  16 \
        --max_len 128 \
        --max_len_templates 32 \
        --dropout 0.3 \
        --learning_rate 1e-5 \
        --alpha 0.2 \
        --beta 0.7 \
        --gamma 0.5 \
        --temperature1 0.5 \
        --temperature2 0.5 \
        --epochs 5 \
        --num_routing 3 \
        --output_dim 2 \
        --output_atoms 100

ran=$((1000 + RANDOM % 9000))
CUDA_VISIBLE_DEVICES=3 python -u run_eRisk2018.py \
          --seed $ran \
          --input_dir "../../data/depression/eRisk2018/processed/phq9_temp2_train_val_test_maxsim16" \
          --save_path "../../data/depression/eRisk2018/result/model_$ran.pth" \
          --save_log_path "../../data/depression/eRisk2018/result/result.txt" \
          --model_type "bert-base-uncased" \
          --train_batch_size 16 \
          --test_batch_size  16 \
          --max_len 128 \
          --max_len_templates 32 \
          --dropout 0.3 \
          --learning_rate 2e-5 \
          --alpha 0.2 \
          --beta 0.4 \
          --gamma 0.3 \
          --temperature1 0.5 \
          --temperature2 0.5 \
          --epochs 15 \
          --num_routing 3 \
          --output_dim 2 \
          --output_atoms 50