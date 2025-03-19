import functools
import shutil
from io import StringIO
from pathlib import Path
from typing import List, Optional, Sequence

import yaml
from deli import load, save
from lazycon import Config
from lightning import LightningModule, Trainer
from lightning_utilities.core.rank_zero import rank_zero_info
from typer import Abort, Argument, Option
from typing_extensions import Annotated

from ..config import log_hyperparam
from ..layout import Layout, Node, Single
from ..pydantic_compat import model_validate
from ..torch.utils import last_checkpoint
from ..utils import chdir
from .app import app
from .backend import BackendCommand


ExpArg = Annotated[Path, Argument(show_default=False, help='Path to the experiment.')]
ConfArg = Annotated[Path, Argument(show_default=False, help='The config from which the experiment will be built.')]
UpdArg = Annotated[List[str], Option(
    ..., '--update', '-u', help='Overwrite specific config entries.', show_default=False
)]
OverwriteArg = Annotated[bool, Option("--overwrite", "-o", help="If specified, overwrites target directory.")]
NamesArg = Annotated[Optional[str], Option(..., help='Names of sub-experiments to start.')]


@app.command()
def start(
        experiment: ExpArg,
        name: Annotated[Optional[str], Argument(help='The name of the sub-experiment to start')] = None,
):
    """ Start a part of an experiment. Mainly used as an internal entrypoint for other commands. """
    experiment = Path(experiment)
    if not experiment.is_absolute():
        print('The `experiment` argument must be an absolute.')
        raise Abort(1)

    config_path = experiment / 'experiment.config'
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

    # execute pre-run callbacks
    if not main_config.get("CALLBACKS", None):
        rank_zero_info("THUNDER: No pre-run callbacks were executed.")

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
        names = set(config) - {"module", "trainer", "train_data", "val_data", "ExpName", "GroupName", "datamodule"}
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

        ckpt_path = last_checkpoint(".")

        if "datamodule" in config:
            trainer.fit(module, datamodule=config.datamodule, ckpt_path=ckpt_path)
            new_ckpt_path = last_checkpoint(".")
            trainer.test(module, datamodule=config.datamodule, ckpt_path=new_ckpt_path)
            trainer.predict(module, datamodule=config.datamodule, ckpt_path=new_ckpt_path)
        else:
            trainer.fit(module, config.train_data, config.get('val_data', None), ckpt_path=ckpt_path)
            new_ckpt_path = last_checkpoint(".")
            if "test_data" in config:
                trainer.test(module, config.test_data, ckpt_path=new_ckpt_path)
            if "predict_data" in config:
                trainer.predict(module, config.predict_data, ckpt_path=new_ckpt_path)


@app.command()
def build(
        config: ConfArg,
        experiment: ExpArg,
        overwrite: OverwriteArg = False,
        update: UpdArg = (),
):
    """ Build an experiment. """
    updates = {}
    for upd in update:
        # TODO: raise
        name, value = upd.split('=', 1)
        updates[name] = yaml.safe_load(StringIO(value))

    experiment = Path(experiment)
    if experiment.exists():
        if overwrite:
            shutil.rmtree(experiment)
        else:
            print(f'Cannot create an experiment in the folder "{experiment}", it already exists. '
                  'If you want to overwrite it, use --overwrite / -o flag.')
            raise Abort(1)

    build_exp(Config.load(config), experiment, updates)


def build_exp(config, experiment, updates):
    experiment = Path(experiment)
    new = set(updates) - set(config)
    if new:
        raise ValueError(f'The names {new} are missing from the config')
    if updates:
        config = config.update(**updates)

    layout: Layout = config.get('layout', Single())
    # TODO: permissions
    experiment.mkdir(parents=True)
    try:
        # build the layout
        # TODO: check name uniqueness
        nodes = list(layout.build(experiment, config))
        if nodes:
            save([node.dict() for node in nodes], experiment / 'nodes.json')

    except Exception:
        shutil.rmtree(experiment)
        raise


@app.command(cls=BackendCommand)
def run(
        experiment: ExpArg,
        names: NamesArg = None,
        *,
        backend,
        **kwargs,
):
    """ Run a built experiment using a given backend. """
    if names is not None:
        names = names.split(',')
    backend, config = BackendCommand.get_backend(backend, kwargs)
    experiment = Path(experiment).absolute()
    if not experiment.exists():
        raise ValueError(f"Trying to run experiment from folder {experiment}, "
                         "but it does not exist.")
    backend.run(config, experiment, get_nodes(experiment, names))


@app.command(cls=BackendCommand)
def build_run(
        config: ConfArg,
        experiment: ExpArg,
        overwrite: OverwriteArg = False,
        update: UpdArg = (),
        names: NamesArg = None,
        *,
        backend,
        **kwargs,
):
    """ A convenient combination of `build` and `run` commands. """
    build(config, experiment, overwrite, update)
    run(experiment, names, backend=backend, **kwargs)


def load_nodes(experiment: Path):
    nodes = experiment / 'nodes.json'
    if not nodes.exists():
        return {}
    # TODO: check uniqueness
    parse_obj = functools.partial(model_validate, Node)
    return {x.name: x for x in map(parse_obj, load(nodes))}


def get_nodes(experiment: Path, names: Optional[Sequence[str]]):
    nodes = load_nodes(experiment)

    if names is None:
        if nodes:
            return nodes.values()
        return

    return [nodes[x] for x in names]
