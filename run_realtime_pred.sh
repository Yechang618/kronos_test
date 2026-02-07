#!/bin/bash
#$ -S /bin/bash

nohup python src/realtime_prediction_v4.py --kc > realtime_pred.log
