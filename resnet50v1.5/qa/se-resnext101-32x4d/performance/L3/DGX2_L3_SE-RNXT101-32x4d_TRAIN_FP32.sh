python ./qa/testscript.py /imagenet --raport `basename ${0} .sh`_raport.json --workspace $1 $2 -j 3 --data-backends syntetic dali-gpu dali-cpu pytorch --bench-iterations 100 --bench-warmup 3 --epochs 2 --arch se-resnext101-32x4d -c fanin --label-smoothing 0.1 --mixup 0.0 --mode training --ngpus 1 16  --bs 64 48 32