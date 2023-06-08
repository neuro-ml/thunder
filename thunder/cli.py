import shutil
from io import StringIO
from pathlib import Path
from typing import List, Optional, Annotated

import yaml
from lazycon import Config
from lightning import LightningModule, Trainer
from typer import Typer, Option, Argument
from deli import load, save

from .layout import Layout, Single, Node
from .utils import chdir

app = Typer(pretty_exceptions_enable=False)


@app.command()
def start(experiment_or_config: Path = Argument(show_default=False), name: Annotated[Optional[str], Argument()] = None):
    experiment_or_config = Path(experiment_or_config)
    if experiment_or_config.is_dir():
        experiment, config_path = experiment_or_config, experiment_or_config / 'experiment.config'
    else:
        experiment, config_path = experiment_or_config.parent, experiment_or_config

    nodes = {}
    if (experiment / 'nodes.json').exists():
        # TODO: check uniqueness
        nodes = {x.name: x for x in map(Node.parse_obj, load(experiment / 'nodes.json'))}

    if name is None:
        if len(nodes) > 1:
            # TODO
            raise ValueError
        elif len(nodes) == 1:
            node, = nodes.values()
        else:
            node = None
    else:
        node = nodes[name]

    # load the main config
    main_config = Config.load(config_path)
    # get the layout
    # FIXME
    main_layout: Layout = getattr(main_config, 'layout', Single())
    config, root, params = main_layout.load(experiment, node)

    with chdir(root):
        layout: Layout = getattr(config, 'layout', Single())
        layout.set(**params)

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
        trainer.fit(module, config.train_data, getattr(config, 'val_data', None), ckpt_path='last')


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
        layout: Layout = getattr(config, 'layout', Single())
        # TODO: check name uniqueness
        nodes = list(layout.build(experiment, config))
        if nodes:
            save([node.dict() for node in nodes], experiment / 'nodes.json')

    except Exception:
        shutil.rmtree(experiment)
        raise


# @app.command()
# def run(config: Path, experiment: Path):
#     new_config = build(config, experiment)
#     start(new_config)


def main():
    app()
