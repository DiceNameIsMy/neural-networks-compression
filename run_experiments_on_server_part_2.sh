#!/bin/bash

CUDA_VISIBLE_DEVICES=1 nohup time python src/main.py -l debug nas -d breast-cancer --batch-size 32 --epochs 30 --patience 5 --evaluations 3 --generations 5 --population 40 --offspring 8 --store-models -P -H -O reports/breast-cancer &
CUDA_VISIBLE_DEVICES=2 nohup time python src/main.py -l debug nas -d cardio --batch-size 32 --epochs 30 --patience 5 --evaluations 3 --generations 5 --population 40 --offspring 8 --store-models -P -H -O reports/cardio &
CUDA_VISIBLE_DEVICES=3 nohup time python src/main.py -l debug experiment1 --epochs 30 --patience 5 --evaluations 3 --size mini &
