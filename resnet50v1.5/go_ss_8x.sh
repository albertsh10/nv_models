#!/bin/bash

# export CUDA_VISIBLE_DEVICES=4
_EPOCHS=90
_WARMUPS=8
_BS=128
_OBS=`jq -n $_BS*8`
_BASELR=`jq -n $_OBS*0.001`
_DATAPATH="/data/imagenet_pytorch/torch"

_DEBUG=true
# _DEBUG=false

if [ $_DEBUG == true ];
then
	python ./multiproc.py --nproc_per_node 8 ./main.py $_DATAPATH --data-backend pytorch --raport-file raport_stage_1.json -j5 -p 100 --lr $_BASELR --optimizer-batch-size $_OBS --warmup $_WARMUPS --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace ${1:-./} -b $_BS --fp16 --static-loss-scale 128 --epochs $_EPOCHS
else
	nohup python ./multiproc.py --nproc_per_node 8 ./main.py $_DATAPATH --data-backend pytorch --raport-file raport_stage_1.json -j5 -p 100 --lr $_BASELR --optimizer-batch-size $_OBS --warmup $_WARMUPS --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace ${1:-./} -b $_BS --fp16 --static-loss-scale 128 --epochs $_EPOCHS > tmplog 2>&1 &
	nohup tensorboard --logdir=runs --port=6006 > tbtmplog 2>&1 &
fi

