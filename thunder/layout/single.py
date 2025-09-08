from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lazycon import Config

from .interface import Layout, Node


class Single(Layout):
    def build(self, experiment: Path, config: Config) -> Iterable[Node]:
        name = experiment.name
        config = config.copy().update(ExpName=name, GroupName=name)
        config.dump(experiment / "experiment.config")
        return []

    def load(self, experiment: Path, node: Node | None) -> tuple[Config, Path, dict[str, Any]]:
        if node is not None:
            raise ValueError(f"Unknown name: {node.name}")
        return Config.load(experiment / "experiment.config"), experiment, {}

    def set(self):
        pass
