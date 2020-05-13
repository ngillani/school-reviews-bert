"""
Usage:

PYTHONPATH=. python src/models/sweeps/bert_reviews_sweep.py --outcome mn_avg_eb --groupname testsweep

"""

import argparse
from src.models.core.experiments import run_param_sweep

import platform
print(platform.python_version())

CMD = 'PYTHONPATH=. python2 src/models/bert_reviews.py'

NGPUS_PER_RUN = 1

GRID = {
     # Training
    'lr': [
        # 0.0005,
        0.0001,
    ],
    'dropout': [
        # 0.1,
        0.3,
    ],
    'hid_dim': [
        256,
        # 768,
    ],
    'model_type': [
       'meanbert',
	# 'robert --n_layers 1',
#	'robert --n_layers 2'
    ],
    'adv_terms': [
        'perfrl,perwht',
        'share_collegeplus',
        'share_singleparent',
        'totenrl'
    ]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groupname', help='name of subdir to save runs')
    parser.add_argument('--outcome', default='mn_avg_eb')
    parser.add_argument('--email_groupname', default='bert_reviews', help='Sent in email when sweep is completed.')
    args = parser.parse_args()

    GRID['outcome'] = [args.outcome]

    base_cmd = CMD + ' --groupname {}'.format(args.groupname)

    run_param_sweep(base_cmd, GRID, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_nmin=10, check_queue_every_nmin=10,
                    email_groupname=args.email_groupname, free_gpu_max_mem=0.4)
