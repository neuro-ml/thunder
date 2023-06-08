from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence, Annotated, Type

from joblib import Parallel, delayed
from pydantic import BaseModel, Extra, validator
from typer import Option

from .layout import Node


class BackendConfig(BaseModel):
    class Config:
        extra = Extra.ignore


class Backend:
    Config: Type[BackendConfig]

    @staticmethod
    def run(config: BackendConfig, experiment: Path, nodes: Optional[Sequence[Node]]):
        pass


class BackendEntryConfig(BaseModel):
    backend: str
    config: BackendConfig

    @validator('config', pre=True)
    def _val_config(cls, v, values):
        val = backends[values['backend']]
        return val.Config.parse_obj(v)

    @property
    def backend_cls(self):
        return backends[self.backend]

    class Config:
        extra = Extra.ignore


class Cli(Backend):
    class Config(BackendConfig):
        n_workers: Annotated[int, Option(..., help='The number of worker processes to spawn')] = 1

    @staticmethod
    def run(config: Cli.Config, experiment: Path, nodes: Optional[Sequence[Node]]):
        if nodes is None:
            subprocess.check_call(['thunder', 'start', str(experiment)])
            return

        Parallel(backend='loky', n_jobs=config.n_workers)(
            delayed(subprocess.check_call)(['thunder', 'start', str(experiment), node.name])
            for node in nodes
        )


class Slurm(Backend):
    class Config(BackendConfig):
        cpu: Annotated[float, Option(...)] = None
        ram: Annotated[float, Option(...)] = None

    @staticmethod
    def run(config: Slurm.Config, experiment: Path, nodes: Optional[Sequence[Node]]):
        raise NotImplementedError


backends = {
    'cli': Cli,
    'slurm': Slurm,
}
