#!/bin/bash
#$ -S /bin/bash

nohup python finetune_25d/train_predictor.py > train_predict.log
