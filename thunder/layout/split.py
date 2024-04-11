from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from deli import load, save
from lazycon import Config
from more_itertools import zip_equal
from torch.utils.data import Dataset, Subset

from ..utils import collect
from .interface import Layout, Node


try:
    import connectome
except ImportError:
    connectome = None
try:
    from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit

    SplitType = Union[Callable, BaseShuffleSplit, BaseCrossValidator]
except ImportError:
    SplitType = Callable


class Split(Layout):
    def __init__(self, split: SplitType, entries: Sequence, *args: Any, names: Optional[Sequence[str]] = None,
                 **kwargs: Any):
        """
        Splits data according to split function.
        Parameters
        ----------
        split: Callable
            Split function, or a sklearn splitter.
        entries: Sequence
            Series of ids or torch Dataset or Connectome Layer.
        args: Any
            args for split.
        names: Optional[Sequence[str]]
            Names of folds, e.g. 'train', 'val', test'
        kwargs: Any
            kwargs for split.
        Examples
        ----------
        ```python
        from sklearn.model_selection import KFold

        ids = [0, 1, ...]
        layout = Split(KFold(3), ids, names=["train", "test"])
        ```
        """
        if not callable(split):
            if not hasattr(split, 'split'):
                raise TypeError(f'Expected either a function, or a sklearn splitter, got {type(split)!r}')
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
    def __init__(self, entries: Sequence, *, shuffle: bool = True,
                 random_state: Union[np.random.RandomState, int, None] = 0,
                 **sizes: Union[int, float]):
        """
        Creates single fold experiment, with custom number of sets.
        Parameters
        ----------
        entries: Sequence
            Sequence of ids or
        shuffle: bool
            Whether to shuffle entries.
        random_state : Union[np.random.RandomState, int, None]
        sizes: Union[int, float]
            Size of each split.
        Examples
        ----------
        ```python
        ids = [...]
        layout = SingleSplit(ids, train=0.7, val=0.1, test=0.2)
        ```
        """
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        ids = entries_to_ids(entries)
        self.entries = entries
        self.split = dict(zip_equal(sizes.keys(), multi_split(
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
    if connectome is not None and isinstance(entries, connectome.Layer):
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

    if not all(s > 0 for s in sizes):
        raise ValueError(f"All sizes must be non-negative ints and floats, got {sizes}.")

    total_size = sum(sizes)
    if total_size != 1 and isinstance(sizes, float):
        raise ValueError("If sizes are specified as floats, they should sum up to 1, "
                         f"got sum({sizes}) = {total_size}.")
    elif all(isinstance(s, int) for s in sizes) and total_size != total:
        raise ValueError("If sizes are specified as ints, they should sum up to number of cases, "
                         f"got sum({sizes}) = {total_size} and {total} cases.")

    sizes = [round(total * x) if isinstance(x, float) else x for x in sizes]

    start = 0
    for size in sizes[:-1]:
        yield ids[start:start + size]
        start += size
    yield ids[start:]


def jsonify(x):
    if isinstance(x, (np.generic, np.ndarray)):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(map(jsonify, x))
    return x
