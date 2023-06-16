import shutil
from io import StringIO
from pathlib import Path
from typing import List, Optional, Annotated, Type, Sequence

import yaml
from deli import load, save
from lazycon import Config
from lightning import LightningModule, Trainer
from typer import Typer, Option, Argument

from ..backend import Backend
from ..config import log_hyperparam
from ..layout import Layout, Single, Node
from ..utils import chdir

app = Typer(name='thunder', pretty_exceptions_enable=False)
ExpArg = Annotated[Path, Argument(show_default=False, help='Path to the experiment')]
ConfArg = Annotated[Path, Argument(show_default=False, help='The config from which the experiment will be built')]
UpdArg = Annotated[List[str], Option(
    ..., '--update', '-u', help='Overwrite specific config entries', show_default=False
)]
NamesArg = Annotated[Optional[str], Option(..., help='Names of sub-experiments to start')]


@app.command()
def start(
        experiment_or_config: ExpArg,
        name: Annotated[Optional[str], Argument(help='The name of the sub-experiment to start')] = None
):
    """ Start a part of an experiment. Mainly used as an internal entrypoint for other commands. """
    experiment_or_config = Path(experiment_or_config)
    if experiment_or_config.is_dir():
        experiment, config_path = experiment_or_config, experiment_or_config / 'experiment.config'
    else:
        experiment, config_path = experiment_or_config.parent, experiment_or_config

    nodes = load_nodes(experiment)

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
    main_layout: Layout = main_config.get('layout', Single())
    config, root, params = main_layout.load(experiment, node)

    with chdir(root):
        layout: Layout = config.get('layout', Single())
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
            else:
                log_hyperparam(trainer.logger, name, value)

        if hyperparams:
            trainer.logger.log_hyperparams(hyperparams)

        trainer.fit(module, config.train_data, config.get('val_data', None), ckpt_path='last')
        if 'test_data' in config:
            trainer.test(module, config.test_data, ckpt_path='last')


@app.command()
def build(
        config: ConfArg,
        experiment: ExpArg,
        update: UpdArg = (),
):
    """ Build an experiment """
    updates = {}
    for upd in update:
        # TODO: raise
        name, value = upd.split('=', 1)
        updates[name] = yaml.safe_load(StringIO(value))

    build_exp(config, experiment, updates)


def build_exp(config, experiment, updates):
    config = Config.load(config)
    experiment = Path(experiment)
    new = set(updates) - set(config)
    if new:
        raise ValueError(f'The names {new} are missing from the config')
    if updates:
        config = config.update(**updates)

    # TODO: permissions
    experiment.mkdir(parents=True)
    try:
        # build the layout
        layout: Layout = config.get('layout', Single())
        # TODO: check name uniqueness
        nodes = list(layout.build(experiment, config))
        if nodes:
            save([node.dict() for node in nodes], experiment / 'nodes.json')

    except Exception:
        shutil.rmtree(experiment)
        raise


def run(
        experiment: ExpArg,
        names: NamesArg = None,
        *,
        backend: Type[Backend],
        **kwargs,
):
    """ Run a built experiment using a given backend. """
    if names is not None:
        names = names.split(',')
    config = backend.Config(**kwargs)
    backend.run(config, experiment, get_nodes(experiment, names))


def build_run(
        config: ConfArg,
        experiment: ExpArg,
        update: UpdArg = (),
        names: NamesArg = None,
        *,
        backend: Type[Backend],
        **kwargs,
):
    """ A convenient combination of `build` and `run` commands. """
    build(config, experiment, update)
    run(experiment, names, backend=backend, **kwargs)


def load_nodes(experiment: Path):
    nodes = experiment / 'nodes.json'
    if not nodes.exists():
        return {}
    # TODO: check uniqueness
    return {x.name: x for x in map(Node.parse_obj, load(nodes))}


def get_nodes(experiment: Path, names: Optional[Sequence[str]]):
    nodes = load_nodes(experiment)

    if names is None:
        if nodes:
            return nodes.values()
        return

    return [nodes[x] for x in names]