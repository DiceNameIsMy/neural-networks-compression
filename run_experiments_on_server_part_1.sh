#!/bin/bash

CUDA_VISIBLE_DEVICES=1 nohup time python src/main.py -l debug nas -d vertebral --batch-size 32 --epochs 30 --patience 5 --evaluations 3 --generations 5 --population 40 --offspring 8 --store-models -P -H -O reports/vertebral &
CUDA_VISIBLE_DEVICES=2 nohup time python src/main.py -l debug nas -d mini-mnist --batch-size 50 --epochs 30 --patience 5 --evaluations 3 --generations 5 --population 40 --offspring 8 --store-models -P -H -O reports/mini-mnist &
CUDA_VISIBLE_DEVICES=3 nohup time python src/main.py -l debug nas -d mini-cifar10 --batch-size 100 --epochs 30 --patience 5 --evaluations 3 --generations 5 --population 40 --offspring 8 --store-models -P -H -O reports/mini-cifar10 &
