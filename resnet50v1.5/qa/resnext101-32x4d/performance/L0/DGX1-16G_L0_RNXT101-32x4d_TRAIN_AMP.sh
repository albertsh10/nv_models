python ./qa/testscript.py /imagenet --raport `basename ${0} .sh`_raport.json --workspace $1 $2 -j 5 --data-backends syntetic --bench-iterations 100 --bench-warmup 3 --epochs 1 --arch resnext101-32x4d -c fanin --label-smoothing 0.1 --mixup 0.0 --mode training --ngpus 8 --amp --static-loss-scale 128 --bs 128