#!/bin/bash

idx=${1:-0}
echo $idx
mkdir -p "pretrain_checkpoints$idx"
mv runs/ pretrain_checkpoints$idx/
mv model_best.pth.tar pretrain_checkpoints$idx/
mv checkpoint.pth.tar pretrain_checkpoints$idx/
mv raport_stage_1.json pretrain_checkpoints$idx/

