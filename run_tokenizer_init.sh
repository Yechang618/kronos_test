#!/bin/bash
#$ -S /bin/bash

nohup python finetune/train_tokenizer_init.py > train_tokenizer.log
