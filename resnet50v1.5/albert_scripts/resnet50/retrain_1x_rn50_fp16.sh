#!/bin/bash

# export CUDA_VISIBLE_DEVICES=4
_EPOCHS=${1:-90}
_RET_EPOCH=${2:-0}
_WARMUPS=8
_BS=128 # 803 ips
_BASELR=`jq -n $_BS*0.001`
_DATAPATH="/data/imagenet_pytorch/torch"

_DEBUG=true
# _DEBUG=false

if [ $_DEBUG == true ];
then
        python ./main.py $_DATAPATH --data-backend pytorch --raport-file raport.json -j5 -p 100 --lr $_BASELR --optimizer-batch-size $_BS --warmup $_WARMUPS --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace ${3:-./} -b $_BS --fp16 --static-loss-scale 128 --epochs $_EPOCHS --resume "./model_best.pth.tar" --sparse --retrain_init_epoch $_RET_EPOCH
else
        nohup python ./main.py $_DATAPATH --data-backend pytorch --raport-file raport.json -j5 -p 100 --lr $_BASELR --optimizer-batch-size $_BS --warmup $_WARMUPS --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace ${3:-./} -b $_BS --fp16 --static-loss-scale 128 --epochs $_EPOCHS  --resume "./model_best.pth.tar" --sparse --retrain_init_epoch $_RET_EPOCH > run.log 2>&1 &
        nohup tensorboard --logdir=runs --port=6006 > tb.log 2>&1 &
fi

