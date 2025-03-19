import logging
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest
from deli import load_text, save_text
from lazycon import Config, load as read_config
from typer.testing import CliRunner

import thunder.cli.backend
import thunder.cli.backend_cli
from thunder.cli.backend import collect_configs, load_backend_configs
from thunder.cli.entrypoint import app
from thunder.utils import chdir


runner = CliRunner()


@pytest.fixture()
def mock_backend(temp_dir):
    backends_yml = temp_dir / "backends.yml"
    thunder.cli.backend.BACKENDS_CONFIG_PATH = backends_yml
    thunder.cli.entrypoint._main.BACKENDS_CONFIG_PATH = backends_yml
    thunder.cli.backend_cli.BACKENDS_CONFIG_PATH = backends_yml

    collect_configs.cache_clear()
    if backends_yml.exists():
        os.remove(backends_yml)

    yield backends_yml

    collect_configs.cache_clear()
    if backends_yml.exists():
        os.remove(backends_yml)


def test_build(temp_dir, mock_backend):
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
        assert re.match('Cannot create an experiment in the folder ".*", it already exists. '
                        'If you want to overwrite it, use --overwrite / -o flag.\n', result.output)

    with cleanup(experiment):
        result = invoke('build', config, experiment, '-u', 'c=3')
        assert result.exit_code != 0, result.output
        assert 'are missing from the config' in str(result.exception)

        result = invoke('build', config, experiment, '-u', 'a=10')
        assert result.exit_code == 0, result.output
        assert Config.load(experiment / 'experiment.config').a == 10

    with cleanup(experiment):
        # language=yaml
        mock_backend.write_text('''
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
        assert 'Missing option' in result.output, result.output

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
    assert result.exit_code == 0, result.output
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

    # custom callback
    collect_configs.cache_clear()
    invoke("backend", "add", "cli_for_test_run", "backend=cli")
    result = invoke("run", "--backend", "cli_for_test_run", experiment)
    assert result.exit_code == 0, result.output


@pytest.mark.timeout(60)
def test_run_callbacks(temp_dir, dumb_config, caplog):
    # we use cli function directly since `runner.invoke`
    # erases stdout and logging info
    from thunder.cli.main import start as cli_start

    experiment = temp_dir / "test_run_callbacks_no_cb"
    experiment.mkdir()
    config = experiment / "experiment.config"
    shutil.copy(dumb_config, config)

    # absolute path
    with caplog.at_level(logging.INFO):
        cli_start(experiment)
    assert any("No pre-run callbacks were executed" in r.message for r in caplog.records), caplog.records

    # Add pre run callbacks

    experiment = temp_dir / "test_run_callbacks"
    experiment.mkdir()
    config = experiment / "experiment.config"
    config_text = "\n".join(["from lightning import seed_everything",
                             load_text(dumb_config), "CALLBACKS=[seed_everything()]"])

    save_text(config_text, config)

    # absolute path
    with caplog.at_level(logging.INFO):
        cli_start(experiment)
    assert any("seed set to" in r.message.lower() for r in caplog.records), caplog.records


def test_backend_add(temp_dir, mock_backend):
    result = invoke("backend", "add", "new_config", "backend=slurm", "ram=100")
    assert result.exit_code == 0 and "new_config" in load_backend_configs()

    result = invoke("backend", "add", "new_config", "backend=slurm", "ram=100")
    assert result.exit_code != 0 and "new_config" in load_backend_configs()

    result = invoke("backend", "add", "new_config", "backend=slurm", "ram=200G", "--force")
    assert result.exit_code == 0 and "new_config" in load_backend_configs()
    assert load_backend_configs()["new_config"].config.ram == "200G"

    invoke("backend", "add", "new_config_2", "backend=slurm", "ram=200", "--force")
    local = load_backend_configs()
    assert "new_config" in local and "new_config_2" in local

    invoke("backend", "add", "new_config_3", "backend=cli", "n_workers=8")
    local = load_backend_configs()
    assert "new_config" in local and "new_config_2" in local
    assert "new_config_3" in local


def test_backend_list(temp_dir, mock_backend):
    # language=yaml
    mock_backend.write_text('''
    a:
        backend: cli
        config:
            n_workers: 1
    b:
        backend: cli
        config:
            n_workers: 2
    meta:
        default: b
    ''')

    assert invoke("backend", "list", "a", "b").exit_code == 0
    assert invoke("backend", "list", "c").exit_code == 0


def test_backend_set(temp_dir, mock_backend):
    result = invoke("backend", "add", "config", "backend=slurm", "ram=100G", "--force")
    assert result.exit_code == 0, result.output
    result = invoke("backend", "set", "config")

    assert result.exit_code == 0, result.output
    local = load_backend_configs()
    assert local[local["meta"].default].config.ram == "100G"


def test_backend_remove(temp_dir, mock_backend):
    assert invoke("backend", "add", "config", "backend=slurm", "ram=100", "--force").exit_code == 0
    result = invoke("backend", "remove", "config")

    assert result.exit_code == 0
    assert not load_backend_configs().keys()


def test_datamodule(temp_dir, dumb_config):
    experiment = temp_dir / "test_datamodule"
    experiment.mkdir()
    config = experiment / "experiment.config"
    config_text = "\n".join([
        "from lightning.pytorch.demos.boring_classes import BoringDataModule",
        load_text(dumb_config),
        "datamodule = BoringDataModule()"
    ])

    save_text(config_text, config)

    result = invoke("run", experiment)
    # TODO: we need something smarter?
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
