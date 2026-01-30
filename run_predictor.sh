#!/bin/bash
#$ -S /bin/bash

nohup python finetune/train_predictor.py > train_predict.log
