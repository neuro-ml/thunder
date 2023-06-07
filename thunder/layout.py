import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union, Sequence

import numpy as np
from dpipe.split import train_val_test_split
from connectome import Layer, Filter
from lazycon import Config
from torch.utils.data import Dataset, Subset
from deli import load, save


class Layout(ABC):
    def prepare(self, experiment: Path):
        self.set(**self.load(experiment))

    @abstractmethod
    def build(self, experiment: Path, config: Config):
        pass

    @abstractmethod
    def load(self, experiment: Path) -> Dict[str, Any]:
        pass

    @abstractmethod
    def set(self, **kwargs):
        pass


class CrossValTest(Layout):
    def __init__(self, entries, *, n_folds: int, val_size: int = 0,
                 random_state: Union[np.random.RandomState, int, None] = 0):
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        if isinstance(entries, Layer):
            ids = entries.ids
        elif isinstance(entries, Dataset):
            ids = list(range(len(entries)))
        else:
            ids = entries

        self.entries = entries
        self.n_folds = n_folds
        self.val_size = val_size
        self.random_state = random_state
        self.splits = train_val_test_split(ids, n_splits=n_folds, val_size=val_size, random_state=random_state)
        self.fold: Optional[int] = None

    def _subset(self, idx):
        ids = self.splits[self.fold][idx]
        if isinstance(self.entries, Layer):
            return self.entries >> Filter.keep(ids)
        if isinstance(self.entries, Dataset):
            return Subset(self.entries, ids)
        return ids

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
        for fold in range(self.n_folds):
            folder = experiment / f'fold_{fold}'
            folder.mkdir()
            save(self.splits[fold], folder / 'split.json')

            local = config.copy().update(ExpName=f'fold_{fold}', GroupName=experiment.name)
            local.dump(folder / 'experiment.config')

    def load(self, experiment: Path) -> Dict[str, Any]:
        return {
            'fold': int(experiment.name.split('_')[-1]),
            'split': load(experiment / 'split.json'),
        }

    def set(self, fold: int, split: Optional[Sequence[Sequence]] = None):
        self.fold = fold
        if split is None:
            warnings.warn('No reference split provided. your result might be inconsistent!', UserWarning)
        else:
            if split != self.splits[fold]:
                # TODO: consistency error?
                raise ValueError
