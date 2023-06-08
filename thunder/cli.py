import inspect
import shutil
import sys
from inspect import Parameter
from io import StringIO
from pathlib import Path
from typing import List, Optional, Annotated, Type, Sequence

import typer
import yaml
from deli import load, save
from lazycon import Config
from lightning import LightningModule, Trainer
from typer import Typer, Option, Argument

from .layout import Layout, Single, Node
from .backend import Cli, Backend, BackendEntryConfig
from .utils import chdir

app = Typer(name='thunder', pretty_exceptions_enable=False)


@app.command()
def start(experiment_or_config: Path = Argument(show_default=False), name: Annotated[Optional[str], Argument()] = None):
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


def _run(
        experiment: Annotated[Path, Argument(show_default=False)],
        names: Annotated[Optional[str], Option(...)] = '',
        *,
        backend: Type[Backend],
        **kwargs,
):
    names = names.split(',') or None
    config = backend.Config(**kwargs)
    backend.run(config, experiment, get_nodes(experiment, names))


def main():
    backend_name = detect_backend()
    local_configs = load_backend_configs()
    default_configs = {
        'cli': BackendEntryConfig(backend='cli', config={}),
        'slurm': BackendEntryConfig(backend='slurm', config={}),
    }

    if backend_name is None:
        if len(local_configs) == 1:
            backend_name, = local_configs
            entry = local_configs[backend_name]
            backend = entry.backend_cls
            config = entry.config

        elif len(local_configs) > 1:
            # TODO
            print('bad configs')
            raise typer.Exit(1)

        else:
            backend_name = 'cli'
            backend = Cli
            config = backend.Config()

    else:
        if backend_name in local_configs:
            entry = local_configs[backend_name]
        else:
            entry = default_configs[backend_name]

        backend = entry.backend_cls
        config = entry.config

    all_params = inspect.signature(_run).parameters
    common_params = list(all_params.values())[:2]

    backend_choices = ','.join(set(local_configs) | set(default_configs))
    backend_params = [Parameter(
        'backend', Parameter.POSITIONAL_OR_KEYWORD, default=backend_name,
        annotation=Annotated[Optional[str], Option(
            ..., help=f'The runner backend to use. Choices: {backend_choices}. Currently using {backend_name}.',
            show_default=False,
        )],
    )]
    for field in backend.Config.__fields__.values():
        backend_params.append(Parameter(
            field.name, Parameter.POSITIONAL_OR_KEYWORD, default=getattr(config, field.name),
            annotation=field.outer_type_,
        ))

    def run(**kwargs):
        """ Specify a backend to view specific parameters """
        assert kwargs.pop('backend') == backend_name, kwargs.pop('backend')
        return _run(**kwargs, backend=backend)

    run.__signature__ = inspect.Signature(common_params + backend_params)
    app.command()(run)
    app()


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


def detect_backend():
    args = sys.argv[1:]
    try:
        index = args.index('--backend')
    except ValueError:
        return None

    if index >= len(args) - 1:
        print('The option "--backend" must have an argument')
        raise typer.Exit(1)

    return args[index + 1]


def load_backend_configs():
    path = Path(typer.get_app_dir(app.info.name)) / 'backends.yml'
    if not path.exists():
        return {}

    with path.open('r') as file:
        local = yaml.safe_load(file)
    if local is None:
        return {}
    # FIXME
    assert isinstance(local, dict), type(local)
    return {k: BackendEntryConfig.parse_obj(v) for k, v in local.items()}
