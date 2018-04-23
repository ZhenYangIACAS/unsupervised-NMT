#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

python train_unsupervised.py -c ./configs/config_generator_train.yaml