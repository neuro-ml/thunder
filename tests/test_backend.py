from pathlib import Path

import pytest
from pytest_subprocess import FakeProcess

from thunder.backend import Cli, Slurm
from thunder.layout import Node


@pytest.mark.parametrize('config,nodes', (
    (Cli.Config(), None),
    (Cli.Config(), (Node(name='a'), Node(name='b'))),
))
def test_cli(fake_process: FakeProcess, temp_dir, config, nodes):
    exp_folder = temp_dir / 'some-exp'
    if nodes is None:
        n_calls = 1
        fake_process.register(['thunder', 'start', str(exp_folder)])
    else:
        n_calls = len(nodes)
        for n in nodes:
            fake_process.register(['thunder', 'start', str(exp_folder), n.name])

    Cli.run(config, exp_folder, nodes)
    assert len(fake_process.calls) == n_calls


@pytest.mark.parametrize('config,args,nodes', (
    (Slurm.Config(), (), None),
    (Slurm.Config(cpu=10), ('--cpus-per-task', '10'), None),
    (Slurm.Config(time='5d'), ('--time', None, '--signal=B:INT@30'), None),
    (Slurm.Config(gpu=3), ('--gpus-per-node', '3'), None),
    (Slurm.Config(), ('--array=1-2',), (Node(name='a'), Node(name='b'))),
    (Slurm.Config(limit=1), ('--array=1-2%1',), (Node(name='a'), Node(name='b'))),
))
def test_slurm(fake_process: FakeProcess, temp_dir, config, args, nodes):
    exp_folder = temp_dir / 'some-exp'
    fake_process.register([
        'sbatch', *(fake_process.any(min=1, max=1) if x is None else x for x in args),
        '--job-name', 'some-exp',
        '--output', fake_process.any(min=1, max=1),
        '--error', fake_process.any(min=2, max=2),
    ])
    Slurm.run(config, exp_folder, nodes)
    assert len(fake_process.calls) == 1
    call, = fake_process.calls
    out, err, cmd = call[-4], call[-2], call[-1]
    assert out == err
    shell = Path(cmd).read_text().splitlines()[-1]
    if nodes is None:
        assert shell == f'thunder start {exp_folder}'
    else:
        assert shell == f'thunder start {exp_folder} ${{__NAME}}'
