#!/bin/bash
# ~/kucoin_project/kronos_test/scripts/start_tokenizer.sh

# 进入项目目录
cd ~/kucoin_project/kronos_test

# 激活虚拟环境
source kronos_test/kronos_env/bin/activate

# 确保 Python 缓冲输出（日志实时可见）
export PYTHONUNBUFFERED=1

# 运行训练脚本
exec python finetune/train_tokenizer.py