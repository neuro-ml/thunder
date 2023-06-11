from typing import Union, Any

import numpy as np
import torch
from lightning import LightningModule
from lightning_utilities import apply_to_collection
from torch import nn

from ..utils import squeeze_first


def get_device(x: Union[torch.Tensor, nn.Module]) -> torch.device:
    if isinstance(x, (torch.Tensor, LightningModule)):
        return x.device
    elif isinstance(x, nn.Module):
        return next(x.parameters()).device

    raise TypeError(f"Can't get device of {type(x)}")


def to_np(*x: Any) -> Any:
    return squeeze_first(apply_to_collection(x, torch.Tensor, tensor2np))


def tensor2np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def maybe_from_np(*x: Any, device: Union[torch.device, str]) -> Any:
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.from_numpy(x).to(device)
    return squeeze_first(apply_to_collection(x, (np.ndarray, np.generic, torch.Tensor), to_tensor))
