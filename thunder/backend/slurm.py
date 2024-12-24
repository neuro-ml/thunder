from __future__ import annotations

import datetime
import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from deli import save
from pytimeparse.timeparse import timeparse
from typer import Option
from typing_extensions import Annotated

from ..layout import Node
from ..pydantic_compat import field_validator
from .interface import Backend, BackendConfig, backends


# TODO: neeeds generalization
ROOT = Path('~/.cache/thunder/slurm').expanduser().resolve()
ROOT_CMDSH = ROOT / 'cmdsh'
ROOT_LOGS = ROOT / 'logs'
ROOT_ARRAYS = ROOT / 'arrays'


class Slurm(Backend):
    class Config(BackendConfig):
        ram: Annotated[Optional[str], Option(
            None, '-r', '--ram', '--mem',
            help='The amount of RAM required per node. Default units are megabytes. '
                 'Different units can be specified using the suffix [K|M|G|T].'
        )] = None
        cpu: Annotated[Optional[int], Option(
            None, ..., '-c', '--cpu', '--cpus-per-task', show_default=False,
            help='Number of CPU cores to allocate. Default to 1'
        )] = None
        gpu: Annotated[Optional[int], Option(
            None, '-g', '--gpu', '--gpus-per-node',
            help='Number of GPUs to allocate'
        )] = None
        partition: Annotated[Optional[str], Option(
            None, '-p', '--partition',
            help='Request a specific partition for the resource allocation'
        )] = None
        nodelist: Annotated[Optional[str], Option(
            None,
            help='Request a specific list of hosts. The list may be specified as a comma-separated '
                 'list of hosts, a range of hosts (host[1-5,7,None] for example).'
        )] = None
        time: Annotated[Optional[str], Option(
            None, '-t', '--time',
            help='Set a limit on the total run time of the job allocation. When the time limit is reached, '
                 'each task in each job step is sent SIGTERM followed by SIGKILL.'
        )] = None
        limit: Annotated[Optional[int], Option(
            None,
            help='Limit the number of jobs that are simultaneously running during the experiment',
        )] = None

        @field_validator("time")
        def val_time(cls, v):
            if v is None:
                return
            return parse_duration(v)

        @field_validator("limit")
        def val_limit(cls, v):
            assert v is None or v > 0, 'The jobs limit, if specified, must be positive'
            return v

    @staticmethod
    def run(config: Slurm.Config, experiment: Path, nodes: Optional[Sequence[Node]], wait: Optional[bool] = None):
        def add_option(arg, value, *suffix):
            if value is not None:
                args.extend((f'--{arg}', str(value)))
                args.extend(suffix)

        ROOT_LOGS.mkdir(exist_ok=True, parents=True)
        ROOT_ARRAYS.mkdir(exist_ok=True, parents=True)
        ROOT_CMDSH.mkdir(exist_ok=True, parents=True)
        # TODO: pass the exp name as argument to `run`
        name = experiment.name
        unique_job_name = get_unique_job_name(name)

        args = ['sbatch']
        if nodes is None or len(nodes) == 0:
            log_file = ROOT_LOGS / f'{unique_job_name}.o%j'
            cmds = [shlex.join(['thunder', 'start', str(experiment)])]

        else:
            array = f'--array=1-{len(nodes)}'
            if config.limit is not None:
                array += f'%{config.limit}'

            args.append(array)
            log_file = ROOT_LOGS / f'{unique_job_name}.o%A.%a'
            exp_list = ROOT_ARRAYS / f'{unique_job_name}.json'
            idx = 0
            # we need a unique name
            while exp_list.exists():
                exp_list = ROOT_ARRAYS / f'{unique_job_name}_{idx}.json'
                idx += 1

            save(sorted(x.name for x in nodes), exp_list)
            cmds = [
                '__NAME=$('
                f'python -c "import sys, json; print(json.load(open(sys.argv[1]))[${{SLURM_ARRAY_TASK_ID}}-1])"'
                f' {shlex.quote(str(exp_list))})',
                f'thunder start {shlex.quote(str(experiment))} ${{__NAME}}',
            ]

        add_option('mem', config.ram)
        add_option('cpus-per-task', config.cpu)
        add_option('gpus-per-node', config.gpu)
        add_option('partition', config.partition)
        add_option('nodelist', config.nodelist)
        add_option('time', config.time, '--signal=B:INT@30')
        add_option('job-name', name)
        add_option('output', log_file)
        add_option('error', log_file)
        if wait:
            args.append('--wait')

        script = ROOT_CMDSH / f'{unique_job_name}_cmd.sh'
        script.write_text('\n'.join(['#!/bin/bash'] + cmds))
        args.append(str(script))
        subprocess.check_call(args, stderr=subprocess.STDOUT)


def get_unique_job_name(job_name_prefix):
    job_name_prefix = (job_name_prefix or 'j') + '-'
    if job_name_prefix[0].isdigit():
        job_name_prefix = 'j-' + job_name_prefix

    timestamp = datetime.datetime.now().strftime('%Y-%b%d-%H-%M-%S').lower()
    job_name = job_name_prefix + timestamp
    job_name = job_name.replace('_', '-')
    job_name = job_name.replace(' ', '-')
    job_name = job_name.replace(':', '-')
    return job_name


TIME_REGEX = re.compile(r'^(\d+-)?(\d{1,2})(:\d{1,2}){1,2}$')


def parse_duration(time):
    if TIME_REGEX.match(time):
        return time

    time = parse_time_string(time)
    time = datetime.timedelta(seconds=time)
    days = time.days
    hours, remainder = divmod(time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{days}-{hours:02}:{minutes:02}:{seconds:02}'


def parse_time_string(time):
    parsed = timeparse(time)
    if parsed is None:
        raise ValueError(f'The time format could not be parsed: {time}')
    return parsed


# TODO: need a registry
backends['slurm'] = Slurm
