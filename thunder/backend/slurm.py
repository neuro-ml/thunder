from __future__ import annotations

import datetime
import os
import subprocess
from pathlib import Path
from typing import Annotated, Optional, Sequence

from joblib import Parallel, delayed
from typer import Option

from ..layout import Node
from .interface import Backend, BackendConfig, backends


ROOT = Path('~/.cache/thunder/slurm').expanduser().resolve()
ROOT_CMDSH = ROOT / 'cmdsh'
ROOT_LOGS = ROOT / 'logs'


class Slurm(Backend):
    class Config(BackendConfig):
        mem: Annotated[float, Option(..., help='Amount of RAM (in GB) to allocate')] = 16
        cpus_per_task: Annotated[float, Option(..., help='Number of CPU cores to allocate')] = 4
        gpus_per_task: Annotated[float, Option(..., help='Number of GPU cards to allocate')] = 0

        n_array_jobs: Annotated[int, Option(..., help='Amount of RAM (in GB) to allocate')] = None
        job_name: Annotated[str, Option(...)] = None
        partition: Annotated[str, Option(...)] = None
        nodelist: Annotated[str, Option(...)] = None

    @staticmethod
    def run(config: Slurm.Config, experiment: Path, nodes: Optional[Sequence[Node]]):
        args = ['sbatch']

        unique_job_name = get_unique_job_name(config.job_name)

        # TODO: neeeds generalization
        ROOT_LOGS.mkdir(exist_ok=True, parents=True)
        log_file = ROOT_LOGS / f'{unique_job_name}.o%j'

        args.extend(
            [
                f'--mem {int(config.mem) * 1024}',
                f'--cpus-per-task {config.cpus_per_task}',
                f'--gpus-per-node {config.gpus_per_task}',
                f'--output {log_file}',
                f'--error {log_file}',
            ]
        )

        if config.job_name:
            args.append(f'--job-name {config.job_name}')

        if config.partition:
            args.append(f'--partition {config.partition}')

        if config.nodelist:
            args.append(f'--nodelist {config.nodelist}')

        cmd_sh = write_sh_script([f'thunder start {str(experiment)}'], f'{unique_job_name}_cmd.sh')

        if config.nodes is None:
            subprocess.check_call(' '.join([*args, cmd_sh]))
            return

        def _generator():
            for node in nodes:
                cmd_sh = write_sh_script(
                    [f'thunder start {str(experiment)} {node.name}'], f'{unique_job_name}-{node}_cmd.sh'
                )
                yield delayed(subprocess.check_call)([' '.join([*args, cmd_sh], node.name)])

        Parallel(backend='loky', n_jobs=config.n_workers)(_generator())


def get_unique_job_name(job_name_prefix):
    if job_name_prefix:
        job_name_prefix += '-'
    else:
        job_name_prefix = 'j-'

    if job_name_prefix[0].isdigit():
        job_name_prefix = 'j-' + job_name_prefix

    timestamp = datetime.datetime.now().strftime('%Y-%b%d-%H-%M-%S').lower()
    job_name = job_name_prefix + timestamp
    # filter names to be able to submit jobs to kubernetes
    job_name = job_name.replace('_', '-')
    job_name = job_name.replace(' ', '-')
    job_name = job_name.replace(':', '-')
    return job_name


def write_sh_script(cmd_list, filename, delimiter='\n'):
    # dir for shell scripts
    # TODO: neeeds generalization
    ROOT_CMDSH.mkdir(exist_ok=True, parents=True)

    cmdsh_file = ROOT_CMDSH / filename
    with open(cmdsh_file, 'w') as sh_file:
        sh_file.write('#!/bin/bash\n')
        for i, cmd in enumerate(cmd_list):
            if i == len(cmd_list) - 1:
                delimiter = '\n'
            sh_file.write('{}{}'.format(cmd, delimiter))

    os.chmod(cmdsh_file, 0o755)
    return str(cmdsh_file)


# TODO: need a registry
backends['slurm'] = Slurm
