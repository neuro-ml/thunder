from __future__ import annotations

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Union, List, Dict, Any, Callable

from more_itertools import zip_equal
from toolz import juxt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class Policy(LRScheduler, metaclass=ABCMeta):
    def __init__(self):
        pass

    def __call__(self, optimizer: Optimizer) -> Policy:
        self.set_optimizer(optimizer)
        return self

    def set_optimizer(self, optimizer: Optimizer) -> None:
        super().__init__(optimizer)

    @abstractmethod
    def get_lr(self) -> List[float]:
        pass

    @abstractmethod
    def state_dict(self, *keys: str) -> Dict[str, Any]:
        keys = (*keys, "optimizer")
        return {key: value for key, value in self.__dict__.items() if key not in keys}

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)


class MappingPolicy(Policy, metaclass=ABCMeta):
    def __init__(self, mapping, lr_init: Union[List[float], float] = 1e-3):
        self.current_mapping = None
        self.mapping = mapping

        self.current_lr_init = None
        self.lr_init = lr_init

        super().__init__()

    def set_optimizer(self, optimizer: Optimizer) -> None:
        self.current_mapping = self.mapping
        if isinstance(self.mapping, dict) or callable(self.mapping):
            self.current_mapping = [deepcopy(self.mapping) for _ in optimizer.param_groups]

        self.current_lr_init = self.lr_init
        if isinstance(self.lr_init, (float, int)):
            self.current_lr_init = [self.lr_init for _ in optimizer.param_groups]

        if len(self.current_mapping) != len(optimizer.param_groups):
            raise ValueError(f"Got {len(self.current_mapping)} mappings and {len(optimizer.param_groups)} param groups")

        if len(self.current_lr_init) != len(optimizer.param_groups):
            raise ValueError(f"Got {len(self.current_lr_init)} lr_init and {len(optimizer.param_groups)} param groups")

        for lr_init, param_group in zip(self.current_lr_init, optimizer.param_groups):
            param_group["lr"] = lr_init

        super().set_optimizer(optimizer)

    def __repr__(self) -> str:
        mapping = self.current_mapping if self.current_mapping else self.mapping
        lr_init = self.current_lr_init if self.current_lr_init is not None else self.lr_init
        return f"{self.__class__.__name__}({mapping=}, {lr_init=})"


class Multiply(MappingPolicy):
    mapping: Union[List[Dict[int, float]], Dict[int, float]]

    def get_lr(self) -> List[float]:
        return [
            param_group["lr"] * mapping.get(self.last_epoch, 1)
            for param_group, mapping in zip_equal(self.optimizer.param_groups, self.current_mapping)
        ]

    def state_dict(self) -> Dict[str, Any]:
        return super().state_dict()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)


class Schedule(MappingPolicy):
    mapping: Union[List[Callable], Callable]

    def get_lr(self) -> List[float]:
        return juxt(self.current_mapping)(self.last_epoch)

    def state_dict(self) -> Dict[str, Any]:
        return super().state_dict("mapping")

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)


class Switch(MappingPolicy):
    mapping: Union[List[Dict[int, float]], Dict[int, float]]

    def get_lr(self) -> List[float]:
        return [
            mapping.get(self.last_epoch, param_group["lr"])
            for param_group, mapping in zip_equal(self.optimizer.param_groups, self.current_mapping)
        ]

    def state_dict(self) -> Dict[str, Any]:
        return super().state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
