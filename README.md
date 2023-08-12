# DeCapsNet-demo
## Platform
Our experiments are conducted on a platform with  Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz and an NVIDIA GeForce RTX 3090 GPU.

## Files
- `eRisk2018/RSDD/TRT`：contain the data processing procedure
    - `preprocess.py`：process the raw text
    - `TextPreProcessor.py`：toolkit for handling url and so on
    - `make_dataset.py`：filter posts
- `dataset.py`：process the dataset
- `model.py`：model file of RSDD and TRT
- `model_eRisk2018.py`：model file of eRisk2018
- `parser_utils.py`：parser parameters
- `run.py`：train the model of RSDD and TRT
- `run_eRisk2018.py`：train the model of eRisk2018 
- `statsitics.py`：analyze information from three datasets

## Running
```shell
# within-dataset setting
ran=$((1000 + RANDOM % 9000))
CUDA_VISIBLE_DEVICES=0 python -u run.py \
          --seed $ran \
          --input_dir "../../data/depression/RSDD/processed/phq9_temp2_top64" \
          --save_path "../../data/depression/RSDD/result/model_$ran.pth" \
          --save_log_path "../../data/depression/RSDD/result/result_2W_4L.txt" \
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


# cross-dataset setting
ran=4820
CUDA_VISIBLE_DEVICES=0 python -u run.py \
          --seed $ran \
          --input_dir "../../data/depression/eRisk2018/processed/phq9_temp2_train_val_test_maxsim16" \
          --save_path "../../data/depression/RSDD/result/model_$ran.pth" \
          --save_log_path "../../data/depression/RSDD/result/result_generalization_2W_4L.txt" \
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
          --output_atoms 100 \
          --only_test 1
```
