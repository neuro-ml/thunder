from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Union, Sequence, Tuple, Callable

import numpy as np
from deli import load, save
from jboc import collect
from lazycon import Config
from torch.utils.data import Dataset, Subset

from .interface import Layout, Node

try:
    import connectome
except ImportError:
    connectome = None


# TODO sklearn
class Split(Layout):
    def __init__(self, split: Callable, entries: Sequence, *args: Any, names: Optional[Sequence[str]] = None,
                 **kwargs: Any):
        if not callable(split):
            if not hasattr(split, 'split'):
                # TODO
                raise TypeError(split)
            split = split.split

        ids = entries_to_ids(entries)
        # TODO: safer way to unify types
        splits = [tuple(map(jsonify, xs)) for xs in split(ids, *args, **kwargs)]
        if names is not None:
            # TODO
            assert len(set(names)) == len(names)
            assert len(splits[0]) == len(names)

        self.entries = entries
        self.splits = splits
        self.names = names
        self.fold: Optional[int] = None

    def __getitem__(self, item: int):
        return self._subset(item)

    def __getattr__(self, name: str):
        if self.names is None:
            raise AttributeError(name)
        return self._subset(self.names.index(name))

    def _subset(self, idx):
        # TODO
        assert self.fold is not None
        return entries_subset(self.entries, self.splits[self.fold][idx])

    def build(self, experiment: Path, config: Config):
        config.dump(experiment / 'experiment.config')
        name = experiment.name
        for fold, split in enumerate(self.splits):
            folder = experiment / f'fold_{fold}'
            folder.mkdir()
            save(split, folder / 'split.json')

            local = config.copy().update(ExpName=f'{name}({fold})', GroupName=name)
            local.dump(folder / 'experiment.config')
            yield Node(name=str(fold))

    def load(self, experiment: Path, node: Optional[Node]) -> Tuple[Config, Path, Dict[str, Any]]:
        folder = experiment / f'fold_{node.name}'
        return Config.load(folder / 'experiment.config'), folder, {
            'fold': int(node.name),
            'split': tuple(load(folder / 'split.json')),
        }

    def set(self, fold: int, split: Optional[Sequence[Sequence]] = None):
        self.fold = fold
        if split is None:
            warnings.warn('No reference split provided. Your results might be inconsistent!', UserWarning)
        else:
            if split != self.splits[fold]:
                # TODO: consistency error?
                raise ValueError


class SingleSplit(Layout):
    def __init__(self, entries, *, shuffle: bool = True, random_state: Union[np.random.RandomState, int, None] = 0,
                 **sizes: Union[int, float]):
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        ids = entries_to_ids(entries)
        self.entries = entries
        self.split = dict(zip(sizes.keys(), multi_split(
            ids, list(sizes.values()), shuffle=shuffle, random_state=random_state
        )))

    def __getattr__(self, name: str):
        if name not in self.split:
            raise AttributeError(name)
        return entries_subset(self.entries, self.split[name])

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
            warnings.warn('No reference split provided. Your results might be inconsistent!', UserWarning)
        else:
            if split != self.split:
                # TODO: consistency error?
                raise ValueError


def entries_to_ids(entries):
    if connectome is not None and isinstance(entries, connectome.Layer):
        return entries.ids
    if isinstance(entries, Dataset):
        return list(range(len(entries)))
    return entries


def entries_subset(entries, ids):
    if isinstance(entries, connectome.Layer):
        return entries >> connectome.Filter.keep(ids)
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


def jsonify(x):
    if isinstance(x, (np.generic, np.ndarray)):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(map(jsonify, x))
    return x
