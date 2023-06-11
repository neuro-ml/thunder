from typing import Union, Any

import numpy as np
import torch
from lightning import LightningModule
from lightning_utilities.core.apply_func import apply_to_collection
from torch import nn


def get_device(x: Union[torch.Tensor, nn.Module]) -> Union[torch.device, None]:
    if isinstance(x, (torch.Tensor, LightningModule)):
        return x.device
    elif isinstance(x, nn.Module):
        return next(x.parameters()).device

    raise ValueError(f"Can't get device of {type(x)}")


def to_np(*x: Any) -> Any:
    return apply_to_collection(x, torch.Tensor, tensor2np)


def tensor2np(x: torch.Tensor) -> np.ndarray:
    return x.data.cpu().numpy()


def maybe_from_np(x: Any, device: Union[torch.device, str]) -> Any:
    def to_tensor(x):
        return torch.from_numpy(x).to(device)
    return apply_to_collection(x, (np.ndarray, np.generic), to_tensor)
