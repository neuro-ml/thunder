import os
from contextlib import contextmanager


@contextmanager
def chdir(folder):
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(cwd)
