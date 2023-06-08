from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union, Sequence, Iterable, Tuple

import numpy as np
from dpipe.split import train_val_test_split
from connectome import Layer, Filter
from lazycon import Config
from pydantic import BaseModel, Extra
from torch.utils.data import Dataset, Subset
from deli import load, save


class Node(BaseModel):
    name: str

    # TODO: no layouts with parents so far
    # parents: Sequence[Node] = ()

    class Config:
        extra = Extra.forbid


class Layout(ABC):
    @abstractmethod
    def build(self, experiment: Path, config: Config) -> Iterable[Node]:
        pass

    @abstractmethod
    def load(self, experiment: Path, node: Optional[Node]) -> Tuple[Config, Path, Dict[str, Any]]:
        pass

    @abstractmethod
    def set(self, **kwargs):
        pass


class Single(Layout):
    def build(self, experiment: Path, config: Config) -> Iterable[Node]:
        config = config.copy().update(ExpName=experiment.name)
        config.dump(experiment / 'experiment.config')
        return []

    def load(self, experiment: Path, node: Optional[Node]) -> Tuple[Config, Path, Dict[str, Any]]:
        if node is not None:
            raise ValueError(f'Unknown name: {node.name}')
        return Config.load(experiment / 'experiment.config'), experiment, {}

    def set(self):
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
