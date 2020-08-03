Welcome!  This repository contains code used for constructing and running the BERT-based models applied to our school reviews analysis.

## Key directories and files
- src/models/base/bert_models.py - BERT models
- src/models/dataset.py - code for data prep
- src/models/bert_reviews.py - code for setting model hyper parameters, computing one forward pass, loss
- src/models/core/train_nn.py - wrapper class that handles training model, early stopping, etc.
- src/models/core/experiments.py - config information for running experiments (e.g. gpu allocation, parsing args, etc)
- src/sweeps/bert_reviews_sweep.py - parameter config for running sweep of experiments

## Useful commands

### Running IG on bert models (after initializing virtual env)
sudo bash
source venv/bin/activate
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3.6 interp/bert_interpret.py

### One run (e.g. for debugging)  (after initializing virtual env)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python2 src/models/bert_reviews.py --groupname 'mn_avg_eb_meanbert' --outcome 'mn_avg_eb'
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python2 src/models/bert_reviews.py --groupname 'mn_avg_eb_robert' --outcome 'mn_avg_eb' --model_type 'robert' --hid_dim 768

### Run tensorboard
tensorboard --logdir=<dir>


### Print runs in sorted order according to validation loss
PYTHONPATH=. python src/models/core/experiments.py -d runs/bert_reviews/Mar11_2020/

```
Looking in: runs/bert_reviews/Mar11_2020/

1.3265: runs/bert_reviews/Mar11_2020/debug/hid_dim_128
```

### Run a sweep
PYTHONPATH=. python src/models/sweeps/bert_reviews_sweep.py --outcome mn_avg_eb --groupname pred_confounds --adv_terms=perwht,perfrl

### Location of runs
runs/bert_reviews/Mar08_20/testadvloss/lr0.01_hiddim64/
