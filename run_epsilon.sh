#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0002 &> remote/eval_logs/resnet_dim_dist2/eps_0.0002.log & 

CUDA_VISIBLE_DEVICES=2,3 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0004 &> remote/eval_logs/resnet_dim_dist2/eps_0.0004.log & 

CUDA_VISIBLE_DEVICES=4,5 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0006 &> remote/eval_logs/resnet_dim_dist2/eps_0.0006.log & 

CUDA_VISIBLE_DEVICES=6,7 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0008 &> remote/eval_logs/resnet_dim_dist2/eps_0.0008.log &

echo "waiting for 0.0002-0.0008"
wait
echo "done waiting"

CUDA_VISIBLE_DEVICES=0,1 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.001 &> remote/eval_logs/resnet_dim_dist2/eps_0.001.log & 

CUDA_VISIBLE_DEVICES=2,3 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0012 &> remote/eval_logs/resnet_dim_dist2/eps_0.0012.log & 

CUDA_VISIBLE_DEVICES=4,5 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0014 &> remote/eval_logs/resnet_dim_dist2/eps_0.0014.log & 

CUDA_VISIBLE_DEVICES=6,7 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0016 &> remote/eval_logs/resnet_dim_dist2/eps_0.0016.log &

echo "waiting for 0.001-0.0016"
wait
echo "done waiting"

CUDA_VISIBLE_DEVICES=0,1 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0018 &> remote/eval_logs/resnet_dim_dist2/eps_0.0018.log & 

CUDA_VISIBLE_DEVICES=2,3 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.002 &> remote/eval_logs/resnet_dim_dist2/eps_0.002.log & 

CUDA_VISIBLE_DEVICES=4,5 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0022 &> remote/eval_logs/resnet_dim_dist2/eps_0.0022.log & 

CUDA_VISIBLE_DEVICES=6,7 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0024 &> remote/eval_logs/resnet_dim_dist2/eps_0.0024.log &

echo "waiting for 0.0018-0.0024"
wait
echo "done waiting"

CUDA_VISIBLE_DEVICES=0,1 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0026 &> remote/eval_logs/resnet_dim_dist2/eps_0.0026.log & 

CUDA_VISIBLE_DEVICES=2,3 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0028 &> remote/eval_logs/resnet_dim_dist2/eps_0.0028.log & 

CUDA_VISIBLE_DEVICES=4,5 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.003 &> remote/eval_logs/resnet_dim_dist2/eps_0.003.log & 

CUDA_VISIBLE_DEVICES=6,7 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0032 &> remote/eval_logs/resnet_dim_dist2/eps_0.0032.log &

echo "waiting for 0.0026-0.0032"
wait
echo "done waiting"

CUDA_VISIBLE_DEVICES=0,1 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0034 &> remote/eval_logs/resnet_dim_dist2/eps_0.0034.log & 

CUDA_VISIBLE_DEVICES=2,3 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0036 &> remote/eval_logs/resnet_dim_dist2/eps_0.0036.log & 

CUDA_VISIBLE_DEVICES=4,5 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.0038 &> remote/eval_logs/resnet_dim_dist2/eps_0.0038.log & 

CUDA_VISIBLE_DEVICES=6,7 python3 error_detect.py --config_path configs/pspnet_full_dim_dist.config --eval_dir remote/eval_logs/resnet_dim_dist2/ --trained_checkpoint remote/train_logs/resnet_dim_dist2/model.ckpt-1263 --global_cov --epsilon 0.004 &> remote/eval_logs/resnet_dim_dist2/eps_0.004.log &

echo "waiting for 0.0034-0.004"
wait
echo "done waiting"