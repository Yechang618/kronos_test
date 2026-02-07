#!/bin/bash
#$ -S /bin/bash

nohup python src/realtime_prediction_v3.py --kc > realtime_pred.log
