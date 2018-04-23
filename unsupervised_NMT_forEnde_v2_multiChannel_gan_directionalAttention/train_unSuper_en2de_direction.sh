#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='8,9,10,11,12,13,14,15'

python train_unsupervised.py -c ./configs/config_generator_train.yaml
