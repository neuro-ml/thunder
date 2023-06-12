from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence, Annotated

from joblib import Parallel, delayed
from typer import Option

from .interface import Backend, BackendConfig, backends
from ..layout import Node


class Slurm(Backend):
    class Config(BackendConfig):
        mem: Annotated[float, Option(...)] = 16
        cpus_per_task: Annotated[float, Option(...)] = 4
        gpus_per_task: Annotated[float, Option(...)] = 0

        job_name: Annotated[str, Option(...)] = None
        partition: Annotated[str, Option(...)] = None
        nodelist: Annotated[str, Option(...)] = None

    @staticmethod
    def run(config: Slurm.Config, experiment: Path, nodes: Optional[Sequence[Node]]):
        args = ['sbatch']

        args.extend(
            [
                f'--mem {int(config.mem)}',
                f'--cpus-per-task {config.cpus_per_task}',
                f'--gpus-per-node {config.gpus_per_task}',
            ]
        )

        if config.job_name:
            args.append(f'--job-name {config.job_name}')

        if config.partition:
            args.append(f'--partition {config.partition}')

        if config.nodelist:
            args.append(f'--nodelist {config.nodelist}')

        args.extend(['/PATH/TO/LAUNCHSCRIPT.sh'])

        if config.nodes is None:
            subprocess.check_call(args)
            return

        Parallel(backend='loky', n_jobs=config.n_workers)(
            delayed(subprocess.check_call)([*args, node.name]) for node in nodes
        )


# TODO: need a registry
backends['slurm'] = Slurm
