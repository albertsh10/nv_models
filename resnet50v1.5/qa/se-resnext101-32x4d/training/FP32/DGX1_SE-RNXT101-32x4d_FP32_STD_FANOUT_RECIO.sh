python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --raport-file raport.json -j5 -p 100 --arch se-resnext101-32x4d --label-smoothing 0.1 --workspace $1 -b 64 --lr 0.4 --mom 0.9 --lr-schedule step --epochs 90 --warmup 5 --wd 0.0001 -c classic --data-backend dali-cpu-recio