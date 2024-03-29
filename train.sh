#!/usr/bin/env bash

python train.py --data_config config/LJSpeech/data.yaml \
    --model_config config/LJSpeech/model.yaml \
    --train_config config/LJSpeech/train.yaml \
    --num_gpus 2