#!/bin/bash
#$ -S /bin/bash

nohup python finetune_25d/train_tokenizer.py > train_token_task8.log
