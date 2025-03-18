from __future__ import annotations

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

from more_itertools import zip_equal
from toolz import juxt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class Policy(LRScheduler, metaclass=ABCMeta):
    """
    Policy base class.
    """
    def __init__(self):
        pass

    def __call__(self, optimizer: Optimizer) -> Policy:
        self.set_optimizer(optimizer)
        return self

    def set_optimizer(self, optimizer: Optimizer) -> None:
        """Assigns optimizer to a scheduler"""
        super().__init__(optimizer)

    @abstractmethod
    def get_lr(self) -> List[float]:
        """
        Computes new value of learning rate.
        Returns
        -------
        List[float]
        """
        pass

    def prepare_state_dict(self, *keys: str) -> Dict[str, Any]:
        """
        Creates state dict of scheduler, excluding optimizer and specified keys.
        Be aware that this method does not save state_dict. And only useful for preparing it.
        Parameters
        ----------
        keys: str
            Names of attributes to be excluded from state_dict

        Returns
        -------
        Dict[str, Any]
        """
        return {key: value for key, value in super().state_dict().items() if key not in keys}


class MappingPolicy(Policy, metaclass=ABCMeta):
    def __init__(self, mapping, lr_init: Union[List[float], float] = 1e-3):
        """
        Base class for policy with mapping. Mapping can be a dict or a function
        (it should also be a list of latter types in case of multiple param groups).
        Mapping is the binding between epoch or step number and learning rate value.
        Parameters
        ----------
        mapping
            Binding of epoch or step number and learning rate.
        lr_init: Union[List[float], float]]
            Initial learning rate for each group of parameters.
        """
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

        for lr_init, param_group in zip_equal(self.current_lr_init, optimizer.param_groups):
            param_group["lr"] = lr_init

        super().set_optimizer(optimizer)

    def __repr__(self) -> str:
        mapping = self.current_mapping if self.current_mapping else self.mapping
        lr_init = self.current_lr_init if self.current_lr_init is not None else self.lr_init
        return f"{self.__class__.__name__}({mapping=}, {lr_init=})"


class Multiply(MappingPolicy):
    """
    Multiplies learning rate value on the specified factor in `mapping`.
    Example:
        ```python
            sch = Multiply({1: 0.1, 4: 0.3})
        ```
        if initial learning rate is 1e-3, learning rate will be: 1e-3, 1e-4, 1e-4, 1e-4, 3-e5, ...

    Parameters
    ----------
    mapping: Union[List[Dict[int, float]], Dict[int, float]]
        Maps epoch to factor, keeping the last value between the epochs.
    lr_init: Union[List[float], float]]
        Initial learning rate for each group of parameters.
    """
    mapping: Union[List[Dict[int, float]], Dict[int, float]]

    def get_lr(self) -> List[float]:
        return [
            param_group["lr"] * mapping.get(self.last_epoch, 1)
            for param_group, mapping in zip_equal(self.optimizer.param_groups, self.current_mapping)
        ]

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)


class Schedule(MappingPolicy):
    """
    Assigns learning rate values received from callable mapping.
    Example:
        ```python
        sch = Schedule(np.cos)
        ```
        lr will have values of np.cos(epoch_number)

    Parameters
    ----------
    mapping: Union[List[Callable], Callable]]
        Maps epoch to value.
    lr_init: Union[List[float], float]]
        Initial learning rate for each group of parameters.
    """
    mapping: Union[List[Callable], Callable]

    def get_lr(self) -> List[float]:
        return juxt(self.current_mapping)(self.last_epoch)

    def state_dict(self) -> Dict[str, Any]:
        return self.prepare_state_dict("mapping", "current_mapping")

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)


class Switch(MappingPolicy):
    """
    Assigns learning rate values received from dict mapping.
    Example:
        ```python
        sch = Switch({0: 1e-4, 2: 1e-10)
        ```
        lr: 1e-4, 1e-4, 1e-10, 1e-10, ...

    Parameters
    ----------
    mapping: Union[List[Dict[int, float]], Dict[int, float]]
        Maps specified epochs to specified values, preserving learning rate between epochs.
    lr_init: Union[List[float], float]]
        Initial learning rate for each group of parameters.
    """
    mapping: Union[List[Dict[int, float]], Dict[int, float]]

    def get_lr(self) -> List[float]:
        return [
            mapping.get(self.last_epoch, param_group["lr"])
            for param_group, mapping in zip_equal(self.optimizer.param_groups, self.current_mapping)
        ]

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
