#!/bin/bash
#$ -S /bin/bash

nohup python finetune/train_tokenizer.py > train_tokenizer.log
