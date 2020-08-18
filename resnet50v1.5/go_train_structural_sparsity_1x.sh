#!/bin/bash

export CUDA_VISIBLE_DEVICES=6
_EPOCHS=2
_WARMUPS=1
_DATAPATH="/home/alg/datasets/imagenet_raw"

nohup python ./main.py $_DATAPATH --data-backend syntetic --raport-file raport_stage_1.json -j5 -p 100 --lr 0.256 --optimizer-batch-size 256 --warmup $_WARMUPS --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace ${1:-./} -b 256 --fp16 --static-loss-scale 128 --epochs $_EPOCHS > runlog_stage1 2>&1 & 

echo 'lalala'

# nohup python ./main.py $_DATAPATH --data-backend syntetic --raport-file raport_stage_2.json -j5 -p 100 --lr 0.256 --optimizer-batch-size 256 --warmup $_WARMUPS --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace ${1:-./} -b 256 --fp16 --resume --sparse --static-loss-scale 128 --epochs $_EPOCHS > runlog_stage2 2>&1 & 

