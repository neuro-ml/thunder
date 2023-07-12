import os
from pathlib import Path

import yaml
from lazycon import Config
from typer import Abort, Argument, Typer
from typing_extensions import Annotated

from .backend import BackendCommand
from .main import ConfArg, NamesArg, build_exp, get_nodes


try:
    import wandb
except ImportError:
    wandb = None

wand_app = Typer(name='wandb', help='Wrapper around W&B commands.')


@wand_app.command(cls=BackendCommand)
def agent(
        sweep: Annotated[str, Argument(help='The sweep id to run')],
        config: ConfArg,
        root: Annotated[Path, Argument(show_default=False, help="Path to the experiments' root folder")],
        names: NamesArg = None,
        *,
        backend,
        **kwargs,
):
    def start():
        local = root / os.environ['WANDB_RUN_ID']
        with open(os.environ['WANDB_SWEEP_PARAM_PATH']) as file:
            updates = yaml.safe_load(file)

        assert updates.pop('wandb_version') == 1
        updates = {k: v.pop('value') for k, v in updates.items()}
        build_exp(config.copy(), local, updates)
        backend.run(cnf, Path(local).absolute(), get_nodes(local, names), wait=True)

    if wandb is None:
        print('To use this command please install wandb: `pip install wandb`')
        raise Abort(1)

    if names is not None:
        names = names.split(',')
    backend, cnf = BackendCommand.get_backend(backend, kwargs)
    config = Config.load(config)
    wandb.agent(sweep, start)
