import os
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
import torch
from lightning import LightningModule
from lightning_utilities import apply_to_collection
from torch import nn

from ..utils import squeeze_first


def get_device(x: Union[torch.Tensor, nn.Module]) -> torch.device:
    """
    Infer device of torch.Tensor or nn.Module instance.
    Parameters
    ----------
    x: Union[torch.Tensor, nn.Module]
    Returns
    -------
    device: torch.device
    """
    if isinstance(x, (torch.Tensor, LightningModule)):
        return x.device
    elif isinstance(x, nn.Module):
        try:
            return next(x.parameters()).device
        except StopIteration as e:
            raise RuntimeError("Can't infer the device, because the module has no parameters") from e

    raise TypeError(f"Can't infer the device of {type(x)}")


def to_np(*x: Any) -> Any:
    """
    Converts collection of tensors into numpy arrays.
    Parameters
    ----------
    *x: Any
    Returns
    ----------
    Collection of numpy arrays
    Examples
    --------
    >>> x, y  # torch.Tensor
    >>> z = to_np(x) # convert to numpy array
    >>> x, y = to_np(x, y) # x and y are now numpy arrays
    >>> x, y, z = to_np(x, y, z) # to_np converts only tensors and does not affect other types
    >>> dict_of_np = to_np(dict_of_tensors) # to_np converts any collection
    """
    return squeeze_first(apply_to_collection(x, torch.Tensor, tensor2np))


def tensor2np(x: torch.Tensor) -> np.ndarray:
    """
    Detaches, moves torch.Tensor to CPU and converts into numpy array.
    Parameters
    ----------
    x: torch.Tensor
    Returns
    -------
    np.ndarray
    """
    return x.detach().cpu().numpy()


def maybe_from_np(*x: Any, device: Union[torch.device, str] = "cpu") -> Any:
    """
    Recursively converts numpy arrays to torch.Tensor.
    Parameters
    ----------
    *x: Any
    device: Union[torch.device, str]
        Device to move to, default is CPU.
    Returns
    -------
    Collection of tensors.
    Examples
    -------
    >>> x, y  # np.ndarray
    >>> z = maybe_from_np(x) # convert to torch.Tensor
    >>> x, y = maybe_from_np(x, y) # x and y are now tensors
    >>> x, y, z = maybe_from_np(x, y, z) # maybe_from_np converts np arrays and tensors and does not affect other types
    >>> dict_of_tensors = to_np(dict_of_np) # maybe_from_np converts any collection
    """

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.from_numpy(x).to(device)

    return squeeze_first(apply_to_collection(x, (np.ndarray, np.generic, torch.Tensor), to_tensor))


def last_checkpoint(root: Union[Path, str]) -> Union[Path, Literal["last"]]:
    """
    Load most fresh last.ckpt file based on time.
    Parameters
    ----------
    root: Union[Path, str]
        Path to folder, where last.ckpt or its symbolic link supposed to be.
    Returns
    -------
    checkpoint_path: Union[Path, str]
        If last.ckpt exists - returns Path to it. Otherwise, returns 'last'.
    """
    checkpoints = []
    for p in Path(root).rglob("*"):
        if p.is_symlink():
            p = p.resolve(strict=False)
        if p.suffix == ".ckpt":
            checkpoints.append(p)

    return max(checkpoints, key=lambda t: os.stat(t).st_mtime, default="last")
