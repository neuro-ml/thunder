import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest
from lazycon import Config, load as read_config
from typer.testing import CliRunner

import thunder.cli.backend
from thunder.cli.backend import BACKENDS_CONFIG_PATH, load_backend_configs
from thunder.cli.entrypoint import app
from thunder.utils import chdir


runner = CliRunner()


def mock_backend(folder: Path) -> Path:
    backends_yml = folder / "backends.yml"
    thunder.cli.backend.BACKENDS_CONFIG_PATH = backends_yml
    thunder.cli.entrypoint._main.BACKENDS_CONFIG_PATH = backends_yml

    if backends_yml.exists():
        os.remove(backends_yml)

    return backends_yml


def test_build(temp_dir):
    experiment = temp_dir / 'exp'
    config = temp_dir / 'x.config'
    # language=Python
    config.write_text('''
from thunder.layout import Single
layout = Single()
a = 1
b = 2
    ''')

    with cleanup(experiment):
        result = invoke('build', config, experiment)
        assert result.exit_code == 0, result.output
        assert experiment.exists()
        assert (experiment / 'experiment.config').exists()
        # TODO: nodes.json

        result = invoke('build', config, experiment)
        assert result.exit_code != 0
        assert re.match('Cannot create an experiment in the folder ".*", it already exists\n', result.output)

    with cleanup(experiment):
        result = invoke('build', config, experiment, '-u', 'c=3')
        assert result.exit_code != 0
        assert 'are missing from the config' in str(result.exception)

        result = invoke('build', config, experiment, '-u', 'a=10')
        assert result.exit_code == 0
        assert Config.load(experiment / 'experiment.config').a == 10

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
        result = invoke('run', experiment)
        assert result.exit_code != 0
        assert 'Missing option' in result.output

        # make sure backend configs don't mess with other commands
        result = invoke('build', config, experiment)
        assert result.exit_code == 0, result.output


def test_build_cleanup(temp_dir):
    experiment = temp_dir / 'exp'
    config = temp_dir / 'x.config'
    config.write_text('layout = None')

    result = invoke('build', config, experiment)
    assert result.exit_code != 0
    assert not experiment.exists()


def test_build_overwrite(temp_dir):
    experiment = temp_dir / 'exp'
    experiment.mkdir()
    (experiment / 'experiment.config').write_text('a = 1')

    config = temp_dir / 'new.config'
    config.write_text('b = 2')

    result = invoke('build', config, experiment, "--overwrite")
    assert result.exit_code == 0
    assert not hasattr(read_config(experiment / "experiment.config"), "a")
    assert read_config(experiment / "experiment.config").b == 2


@pytest.mark.timeout(30)
def test_run(temp_dir, dumb_config):
    experiment = temp_dir / "test_run_exp"
    experiment.mkdir()
    config = experiment / "experiment.config"
    shutil.copy(dumb_config, config)

    # absolute path
    result = invoke("run", experiment)
    assert result.exit_code == 0, result.output

    # relative path
    with chdir(experiment.parent):
        # absolute path
        result = invoke("run", experiment.name)
        assert result.exit_code == 0, result.output


def test_add(temp_dir):
    # mocking cock
    mock_backend(temp_dir)

    result = invoke("add", "new_config", "backend=slurm", "ram=100")
    assert result.exit_code == 0 and "new_config" in load_backend_configs()

    result = invoke("add", "new_config", "backend=slurm", "ram=100")
    assert result.exit_code != 0 and "new_config" in load_backend_configs()

    result = invoke("add", "new_config", "backend=slurm", "ram=200", "--force")
    assert result.exit_code == 0 and "new_config" in load_backend_configs()
    assert load_backend_configs()["new_config"].config.ram == "200G"

    invoke("add", "new_config_2", "backend=slurm", "ram=200", "--force")
    local = load_backend_configs()
    assert "new_config" in local and "new_config_2" in local


def test_show(temp_dir):
    backends_yml = mock_backend(temp_dir)

    # language=yaml
    backends_yml.write_text('''
    a:
        backend: cli
        config:
            n_workers: 1
    b:
        backend: cli
        config:
            n_workers: 2
    _default:
        backend: cli
        config:
            n_workers: 1
    ''')

    assert invoke("show", "a", "b").exit_code == 0
    assert invoke("show", "c").exit_code == 0


def test_set(temp_dir):
    mock_backend(temp_dir)

    invoke("add", "config", "backend=slurm", "ram=100", "--force")
    result = invoke("set", "config")

    assert result.exit_code == 0
    assert load_backend_configs()["_default"].config.ram == "100G"


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
