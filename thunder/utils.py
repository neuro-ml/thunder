import os
import random
from contextlib import contextmanager

import numpy as np
import torch
from more_itertools import make_decorator


@contextmanager
def chdir(folder):
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(cwd)


def squeeze_first(inputs):
    """Remove the first dimension in case it is singleton."""
    if len(inputs) == 1:
        inputs = inputs[0]
    return inputs


collect = make_decorator(list)()


def fix_seed(seed=0xBadCafe):
    """Lightning's `seed_everything` with addition `torch.backends` configurations"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
