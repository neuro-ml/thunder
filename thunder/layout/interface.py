from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lazycon import Config
from pydantic import BaseModel


class NoExtra(BaseModel):
    model_config = {"extra": "forbid"}


class Node(NoExtra):
    name: str

    # TODO: no layouts with parents so far
    # parents: Sequence[Node] = ()


class Layout(ABC):
    @abstractmethod
    def build(self, experiment: Path, config: Config) -> Iterable[Node]:
        pass

    @abstractmethod
    def load(self, experiment: Path, node: Node | None) -> tuple[Config, Path, dict[str, Any]]:
        pass

    @abstractmethod
    def set(self, **kwargs):
        pass
