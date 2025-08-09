from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence

from joblib import Parallel, delayed
from typer import Option
from typing_extensions import Annotated

from ..layout import Node
from .interface import Engine, EngineConfig, engines


class Cli(Engine):
    class Config(EngineConfig):
        n_workers: Annotated[int, Option(..., help="The number of worker processes to spawn")] = 1

    @staticmethod
    def run(config: Cli.Config, experiment: Path, nodes: Optional[Sequence[Node]], wait: Optional[bool] = None):
        if nodes is None:
            subprocess.check_call(["thunder", "start", str(experiment)])
            return

        Parallel(backend="loky", n_jobs=config.n_workers)(
            delayed(subprocess.check_call)(["thunder", "start", str(experiment), node.name]) for node in nodes
        )


engines["cli"] = Cli
