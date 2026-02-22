#!/bin/bash
#$ -S /bin/bash

nohup python finetune/train_predictor_init.py > train_pred_task8.log
