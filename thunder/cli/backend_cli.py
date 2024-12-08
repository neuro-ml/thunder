from typing import List

import yaml
from rich.console import Console
from rich.table import Table
from typer import Abort, Argument, Option, Typer
from typing_extensions import Annotated

from ..backend import MetaEntry
from .backend import BACKENDS_CONFIG_PATH, BackendEntryConfig, load_backend_configs


BackendNameArg = Annotated[str, Argument(
    show_default=False, help="Name of the config from your list of backends."
)]
BackendNamesArg = Annotated[List[str], Argument(
    show_default=False, help="Names of the configs from your list of backends."
)]
BackendParamsArg = Annotated[List[str], Argument(
    show_default=False, help="Parameters of added run config.")]
ForceAddArg = Annotated[bool, Option(
    "--force", "-f", help="Forces overwriting of the same backend in .yml file."
)]

console = Console()
backend_app = Typer(name="backend", help="Commands for managing your backends.")


@backend_app.command(name="set")
def _set(name: BackendNameArg):
    """ Set specified backend from list of available backends as default. """
    local = load_backend_configs()
    if name not in local:
        console.print(f"Specified backend `{name} is not among "
                      f"available configs: {sorted(local)}`")
        raise Abort(1)

    local["meta"] = MetaEntry.parse_obj({"default": name})
    with BACKENDS_CONFIG_PATH.open("w") as stream:
        yaml.safe_dump({k: v.dict() for k, v in local.items()}, stream)


@backend_app.command()
def add(name: BackendNameArg, params: BackendParamsArg, force: ForceAddArg = False):
    """ Add run config to the list of available configs. """
    local = load_backend_configs()
    if name in local and not force:
        console.print(f"Backend `{name}` is already present in {str(BACKENDS_CONFIG_PATH)}. "
                      f"If you want it to be overwritten, add --force flag.")
        raise Abort(1)

    kwargs = dict(map(lambda p: p.split("="), params))
    config = {"backend": kwargs.pop("backend", "cli"), "config": kwargs}

    local.update({name: BackendEntryConfig.parse_obj(config)})
    with BACKENDS_CONFIG_PATH.open("w") as stream:
        yaml.safe_dump({k: v.dict() for k, v in local.items()}, stream)


@backend_app.command()
def remove(name: BackendNameArg):
    """ Delete backend from list. """
    local = load_backend_configs()
    if name not in local:
        console.print(f"Backend `{name}` is not among your configs.")
        raise Abort(1)

    local.pop(name)
    with BACKENDS_CONFIG_PATH.open("w") as stream:
        yaml.safe_dump({k: v.dict() for k, v in local.items()}, stream)


@backend_app.command(name="list")
def _list(names: BackendNamesArg = None):
    """ Show parameters of specified backend(s). """
    local = load_backend_configs()

    table = Table("Name", "Backend", "Parameters",
                  title=f"Configs at {str(BACKENDS_CONFIG_PATH.resolve())}")

    if names is None:
        names = local.copy()

    extra = set(names) - set(local)
    if extra:
        console.print("These names are not among your configs:", extra)

    for name in sorted(set(names if names else local) - extra.union({"meta"})):
        entry = local[name].dict()
        table.add_row(*map(str, [name, entry.get("backend", None), entry.get("config", None)]))

    console.print(table)

    if "meta" in local:
        console.print(f"[italic green]Default is [/italic green]{local['meta'].default}")
