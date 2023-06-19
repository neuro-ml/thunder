import os
from contextlib import contextmanager

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
