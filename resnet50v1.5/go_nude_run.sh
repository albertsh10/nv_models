#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
_EPOCHS=10
_WARMUPS=2
_BS=256 # 803 ips
# _BS=128 # 790 ips delta -13 ips
# l2 logging -11 ips
_BASELR=`jq -n $_BS*0.001`
_DATAPATH="/home/alg/datasets/imagenet_raw"

_DEBUG=true
# _DEBUG=false

if [ $_DEBUG == true ];
then
	python ./main.py $_DATAPATH --data-backend pytorch --raport-file raport_stage_1.json -j5 -p 100 --lr $_BASELR --optimizer-batch-size $_BS --warmup $_WARMUPS --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace ${1:-./} -b $_BS --fp16 --static-loss-scale 128 --epochs $_EPOCHS
else
	nohup python ./main.py $_DATAPATH --data-backend pytorch --raport-file raport_stage_1.json -j5 -p 100 --lr $_BASELR --optimizer-batch-size $_BS --warmup $_WARMUPS --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace ${1:-./} -b $_BS --fp16 --static-loss-scale 128 --epochs $_EPOCHS > tmplog 2>&1 &
	nohup tensorboard --logdir=runs --port=9003 > tbtmplog 2>&1 &
fi

