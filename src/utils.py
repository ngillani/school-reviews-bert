# utils.py

import json
import os
import pickle
from pprint import pprint

from nltk.tokenize import (
    word_tokenize,
)

#################################################
#
# Simple I/O
#
#################################################


def save_file(data, path, verbose=False):
    """Creates intermediate directories if they don't exist."""
    dir = os.path.dirname(path)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    if verbose:
        print('Saving: {}'.format(path))

    _, ext = os.path.splitext(path)
    if ext == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=2)
    elif ext == ".json":
        with open(path, "w") as f:
            json.dump(data, f, indent=4, separators=(",", ": "), sort_keys=True)
            f.write("\n")  # add trailing newline for POSIX compatibility


def load_file(path):
    _, ext = os.path.splitext(path)
    if ext == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif ext == ".json":
        with open(path, "r") as f:
            data = json.load(f)
    return data