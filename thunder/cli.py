import os
import shutil
from io import StringIO
from pathlib import Path
from typing import List

import yaml
from lazycon import Config
from lightning import LightningModule, Trainer
from typer import Typer, Option

from .layout import Layout

app = Typer(pretty_exceptions_enable=False)


@app.command()
def start(experiment_or_config: Path):
    experiment_or_config = Path(experiment_or_config)
    if experiment_or_config.is_dir():
        experiment, config = experiment_or_config, experiment_or_config / 'experiment.config'
    else:
        experiment, config = experiment_or_config.parent, experiment_or_config
    experiment, config = experiment.resolve(), config.resolve()

    cwd = os.getcwd()
    os.chdir(experiment)
    try:
        config = Config.load(config)
        if hasattr(config, 'layout'):
            layout: Layout = config.layout
            # FIXME
            layout.prepare(experiment)

        # TODO: match by type rather than name?
        module: LightningModule = config.module
        trainer: Trainer = config.trainer

        # log hyperparams
        names = set(config) - {'module', 'trainer', 'train_data', 'val_data', 'ExpName', 'GroupName'}
        # TODO: lazily determine the types
        hyperparams = {}
        for name in names:
            value = config[name]
            if isinstance(value, (int, float, bool)):
                hyperparams[name] = value
        if hyperparams:
            trainer.logger.log_hyperparams(hyperparams)

        # train
        # TODO: move get(..., default) to lazycon
        trainer.fit(module, config.train_data, getattr(config, 'val_data', None))

    finally:
        os.chdir(cwd)


@app.command()
def build(
        config: Path,
        experiment: Path,
        update: List[str] = Option((), '--update', '-u', help='The source paths to add', show_default=False),
):
    experiment = Path(experiment)
    config = Config.load(config)
    updates = {}
    for upd in update:
        # TODO: raise
        name, value = upd.split('=', 1)
        updates[name] = yaml.safe_load(StringIO(value))
    new = set(updates) - set(config)
    if new:
        raise ValueError(f'The names {new} are missing from the config')
    if updates:
        config = config.update(**updates)

    # TODO: permissions
    experiment.mkdir(parents=True)
    try:
        # build the layout
        if hasattr(config, 'layout'):
            layout: Layout = config.layout
            layout.build(experiment, config)

        else:
            config = config.copy().update(ExpName=experiment.name)
            config.dump(experiment / 'experiment.config')
    except Exception:
        shutil.rmtree(experiment)
        raise


# @app.command()
# def run(config: Path, experiment: Path):
#     new_config = build(config, experiment)
#     start(new_config)


def main():
    app()
