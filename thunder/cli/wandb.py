import os
from pathlib import Path
from typing import Annotated

import wandb
import yaml
from typer import Typer, Argument

from .main import ConfArg, build_exp, NamesArg, get_nodes

wand_app = Typer(name='wandb', help='Wrapper around W&B commands')


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
        build_exp(config, local, updates)

        cnf = backend.Config(**kwargs)
        backend.run(cnf, local, get_nodes(local, names), wait=True)

    if names is not None:
        names = names.split(',')
    wandb.agent(sweep, start)