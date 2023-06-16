from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Union, Sequence, Tuple

import numpy as np
from deli import load, save
from dpipe.split import train_val_test_split
from lazycon import Config

from .interface import Layout, Node
from .split import entries_to_ids, entries_subset


class CrossValTest(Layout):
    def __init__(self, entries, *, n_folds: int, val_size: int = 0,
                 random_state: Union[np.random.RandomState, int, None] = 0):
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        ids = entries_to_ids(entries)
        self.entries = entries
        self.n_folds = n_folds
        self.val_size = val_size
        self.random_state = random_state
        self.splits = train_val_test_split(ids, n_splits=n_folds, val_size=val_size, random_state=random_state)
        self.fold: Optional[int] = None

    def _subset(self, idx):
        return entries_subset(self.entries, self.splits[self.fold][idx])

    @property
    def train(self):
        return self._subset(0)

    @property
    def val(self):
        return self._subset(1)

    @property
    def test(self):
        return self._subset(2)

    def build(self, experiment: Path, config: Config):
        config.dump(experiment / 'experiment.config')
        name = experiment.name
        for fold in range(self.n_folds):
            folder = experiment / f'fold_{fold}'
            folder.mkdir()
            save(self.splits[fold], folder / 'split.json')

            local = config.copy().update(ExpName=f'{name}({fold})', GroupName=name)
            local.dump(folder / 'experiment.config')
            yield Node(name=str(fold))

    def load(self, experiment: Path, node: Optional[Node]) -> Tuple[Config, Path, Dict[str, Any]]:
        folder = experiment / f'fold_{node.name}'
        return Config.load(folder / 'experiment.config'), folder, {
            'fold': int(node.name),
            'split': load(folder / 'split.json'),
        }

    def set(self, fold: int, split: Optional[Sequence[Sequence]] = None):
        self.fold = fold
        if split is None:
            warnings.warn('No reference split provided. your result might be inconsistent!', UserWarning)
        else:
            if split != self.splits[fold]:
                # TODO: consistency error?
                raise ValueError
