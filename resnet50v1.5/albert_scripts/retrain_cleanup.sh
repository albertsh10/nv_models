#!/bin/bash

idx=${1:-0}
echo $idx
mkdir -p "retrain_checkpoints$idx"
mv runs/ retrain_checkpoints$idx/
mv model_best.pth.tar retrain_checkpoints$idx/
mv checkpoint.pth.tar retrain_checkpoints$idx/
mv raport.json retrain_checkpoints$idx/

