#! /bin/bash
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

nvidia-smi

RESULTS_DIR='/results'
CHECKPOINTS_DIR='/results/checkpoints'
STAT_FILE=${RESULTS_DIR}/run_log.json
mkdir -p $CHECKPOINTS_DIR
cp -r /workspace/translation ${RESULTS_DIR}/code
PREC=${1:-'amp'}
SEED=${2:-1}

if [ "$PREC" = "amp" ];
then
    PREC='--amp --amp-level O2 '
    BATCH_SIZE=5120
else
    PREC=''
    BATCH_SIZE=2560
fi



python -m torch.distributed.launch --nproc_per_node 8 /workspace/translation/train.py \
  /data/wmt14_en_de_joined_dict \
  --arch transformer_wmt_en_de_big_t2t \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.997)' \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0006\
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens ${BATCH_SIZE} \
  --seed ${SEED} \
  --max-epoch 40 \
  --online-eval \
  --no-epoch-checkpoints \
  --log-interval 500 \
  --save-dir ${RESULTS_DIR} \
  --stat-file ${STAT_FILE} \
  --distributed-init-method env:// \
  ${PREC}

python scripts/draw_summary.py ${STAT_FILE} ${RESULTS_DIR}/summary.png
