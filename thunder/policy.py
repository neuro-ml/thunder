from abc import ABCMeta, abstractmethod
from typing import Union, List, Dict, Any, Callable

from more_itertools import zip_equal
from toolz import juxt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class Policy(LRScheduler, metaclass=ABCMeta):
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, optimizer: Optimizer) -> "Policy":
        self.set_optimizer(optimizer)
        return self

    def set_optimizer(self, optimizer: Optimizer) -> None:
        if isinstance(self.mapping, dict):
            self.mapping = [self.mapping.copy() for _ in optimizer.param_groups]

        if len(self.mapping) != len(optimizer.param_groups):
            raise ValueError(f"Got {len(self.mapping)} mappings and {len(optimizer.param_groups)} param groups")
        super().__init__(optimizer)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(mapping={self.mapping})"

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def get_lr(self) -> List[float]:
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


class Multiply(Policy):
    def __init__(self, mapping: Union[List[Dict[int, float]], Dict[int, float]]):
        super().__init__(mapping)

    def get_lr(self) -> List[float]:
        return [
            param_group["lr"] * mapping.get(self.last_epoch, 1)
            for param_group, mapping in zip_equal(self.optimizer.param_groups, self.mapping)
        ]

    def state_dict(self) -> Dict[str, Any]:
        state_dict = {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        state_dict['mapping'] = self.mapping
        return state_dict

    def load_state_dict(self, state_dict):
        self.mapping = state_dict.pop('mapping')
        self.__dict__.update(state_dict)


class Schedule(Policy):
    def __init__(self, mapping: Union[Callable, List[Callable]]):
        super().__init__(mapping)

    def get_lr(self) -> List[float]:
        return juxt(self.mapping)(self.last_epoch)

    def state_dict(self) -> Dict[str, Any]:
        return super().state_dict()
