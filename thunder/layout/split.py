from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Union, Sequence, Tuple

import numpy as np
from connectome import Layer, Filter
from deli import load, save
from jboc import collect
from lazycon import Config
from torch.utils.data import Dataset, Subset

from .interface import Layout, Node


class MultiSplit(Layout):
    def __init__(self, entries, *, shuffle: bool = True, random_state: Union[np.random.RandomState, int, None] = 0,
                 **sizes: Union[int, float]):
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        ids = entries_to_ids(entries)
        self.entries = entries
        self.split = dict(zip(sizes.keys(), multi_split(
            ids, list(sizes.values()), shuffle=shuffle, random_state=random_state
        )))

    def build(self, experiment: Path, config: Config):
        config.dump(experiment / 'experiment.config')
        name = experiment.name
        save(self.split, experiment / 'split.json')

        local = config.copy().update(ExpName=name, GroupName=name)
        local.dump(experiment / 'experiment.config')
        return []

    def load(self, experiment: Path, node: Optional[Node]) -> Tuple[Config, Path, Dict[str, Any]]:
        return Config.load(experiment / 'experiment.config'), experiment, {
            'split': load(experiment / 'split.json'),
        }

    def set(self, split: Optional[Dict[str, Sequence]] = None):
        if split is None:
            warnings.warn('No reference split provided. your result might be inconsistent!', UserWarning)
        else:
            if split != self.split:
                # TODO: consistency error?
                raise ValueError

        for name, ids in self.split.items():
            # TODO: unsafe?
            setattr(self, name, entries_subset(self.entries, ids))


def entries_to_ids(entries):
    if isinstance(entries, Layer):
        return entries.ids
    if isinstance(entries, Dataset):
        return list(range(len(entries)))
    return entries


def entries_subset(entries, ids):
    if isinstance(entries, Layer):
        return entries >> Filter.keep(ids)
    if isinstance(entries, Dataset):
        return Subset(entries, ids)
    return ids


@collect
def multi_split(ids: Sequence, sizes: Sequence[int, float],
                shuffle: bool = True, random_state: Union[np.random.RandomState, int, None] = 0):
    if shuffle:
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        # permutation(ids) could change the type of the elements
        ids = [ids[i] for i in random_state.permutation(len(ids))]

    total = len(ids)
    sizes = [round(total * x) if isinstance(x, float) else x for x in sizes]
    negative = [x for x in sizes if x < 0]
    pos = sum(x for x in sizes if x >= 0)
    if len(negative) > 1:
        # TODO
        raise ValueError
    if pos > total:
        raise ValueError

    if len(negative) == 1:
        sizes = [x if x >= 0 else total - pos for x in sizes]

    final = sum(sizes)
    if final != total:
        raise ValueError

    start = 0
    for size in sizes:
        yield ids[start:start + size]
        start += size
