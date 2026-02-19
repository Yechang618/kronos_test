#!/bin/bash
#$ -S /bin/bash

nohup python finetune_25d/train_tokenizer.py > train_tokenizer.log
