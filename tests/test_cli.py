import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

from lazycon import Config
from typer.testing import CliRunner

from thunder.cli.entrypoint import app
from thunder.cli.backend import BACKENDS_CONFIG_PATH

runner = CliRunner()


def test_build(temp_dir):
    experiment = temp_dir / 'exp'
    config = temp_dir / 'x.config'
    # language=Python
    Config.loads('''
from thunder.layout import Single
layout = Single()
    ''').dump(config)

    with cleanup(experiment):
        result = invoke('build', str(config), str(experiment))
        assert result.exit_code == 0, result.output
        assert experiment.exists()
        assert (experiment / 'experiment.config').exists()
        # TODO: nodes.json

        result = invoke('build', str(config), str(experiment))
        assert result.exit_code != 0
        assert re.match('Cannot create an experiment in the folder ".*", it already exists\n', result.output)

    # FIXME: this part will mess with user's local config!
    with cleanup(experiment, BACKENDS_CONFIG_PATH):
        BACKENDS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # language=yaml
        BACKENDS_CONFIG_PATH.write_text('''
a:
    backend: cli
    config:
        n_workers: 1
b:
    backend: cli
    config:
        n_workers: 2
        ''')
        # we don't know which backend to choose
        result = invoke('run', str(experiment))
        assert result.exit_code != 0
        assert 'Missing option' in result.output

        # make sure backend configs don't mess with other commands
        result = invoke('build', str(config), str(experiment))
        assert result.exit_code == 0, result.output


def test_run(temp_dir, dumb_config):
    experiment = temp_dir / "test_run_exp"
    experiment.mkdir()
    config = experiment / "experiment.config"
    Config.load(dumb_config).dump(config)

    result = invoke("run", str(config))
    assert result.exit_code == 0, result.output


def invoke(*cmd):
    return runner.invoke(app, list(map(str, cmd)))


# TODO: restore to previous state
@contextmanager
def cleanup(*paths):
    paths = list(map(Path, paths))
    try:
        yield
    finally:
        for path in paths:
            if not path.exists():
                continue

            if path.is_dir():
                shutil.rmtree(path)
            else:
                os.remove(path)
