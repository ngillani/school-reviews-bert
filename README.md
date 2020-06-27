## Running IG on bert models (after initializing virtual env)
sudo CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3.6 interp/bert_interpret.py


## One run (e.g. for debugging)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python src/models/bert_reviews.py --hid_dim 128

## Run tensorboard
tensorboard --logdir=<dir>


## Print runs in sorted order according to validation loss
PYTHONPATH=. python src/models/core/experiments.py -d runs/bert_reviews/Mar11_2020/

```
Looking in: runs/bert_reviews/Mar11_2020/

1.3265: runs/bert_reviews/Mar11_2020/debug/hid_dim_128
```

## Run a sweep
PYTHONPATH=. python src/models/sweeps/bert_reviews_sweep.py --outcome mn_avg_eb --groupname limited_five_star

## Location of runs
runs/bert_reviews/Mar08_20/testadvloss/lr0.01_hiddim64/
