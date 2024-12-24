from typing import Dict, List, Union

import yaml
from rich.console import Console
from rich.table import Table
from typer import Abort, Argument, Option, Typer
from typing_extensions import Annotated

from ..backend import MetaEntry
from ..pydantic_compat import model_dump, model_validate
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
CreateYMLArg = Annotated[bool, Option(
    "--create", "-c", help="Creates .yml file with stored backends."
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

    local["meta"] = model_validate(MetaEntry, {"default": name})
    with BACKENDS_CONFIG_PATH.open("w") as stream:
        yaml.safe_dump({k: _dump_backend_entry(v) for k, v in local.items()}, stream)


@backend_app.command()
def add(name: BackendNameArg, params: BackendParamsArg, force: ForceAddArg = False,
        create_yml: CreateYMLArg = False):
    """ Add run config to the list of available configs. """
    local = load_backend_configs()
    if name in local and not force:
        console.print(f"Backend `{name}` is already present in {str(BACKENDS_CONFIG_PATH)}. "
                      f"If you want it to be overwritten, add --force flag.")
        raise Abort(1)

    kwargs = dict(map(lambda p: p.split("="), params))
    config = {"backend": kwargs.pop("backend", "cli"), "config": kwargs}

    local.update({name: model_validate(BackendEntryConfig, config)})

    if not BACKENDS_CONFIG_PATH.parent.exists() and not create_yml:
        path = str(BACKENDS_CONFIG_PATH)
        console.print(f"Backends storage {path} does not exist, "
                      f"you can create it by adding --create to the command.")
        raise Abort(1)

    if create_yml:
        BACKENDS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with BACKENDS_CONFIG_PATH.open("w") as stream:
        yaml.safe_dump({k: _dump_backend_entry(v) for k, v in local.items()}, stream)


@backend_app.command()
def remove(name: BackendNameArg):
    """ Delete backend from list. """
    local = load_backend_configs()
    if name not in local:
        console.print(f"Backend `{name}` is not among your configs.")
        raise Abort(1)

    local.pop(name)
    with BACKENDS_CONFIG_PATH.open("w") as stream:
        yaml.safe_dump({k: _dump_backend_entry(v) for k, v in local.items()}, stream)


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
        entry = _dump_backend_entry(local[name])
        table.add_row(*map(str, [name, entry.get("backend", None), entry.get("config", None)]))

    console.print(table)

    if "meta" in local:
        console.print(f"[italic green]Default is [/italic green]{local['meta'].default}")


def _dump_backend_entry(backend: BackendEntryConfig) -> Dict[str, Union[str, Dict]]:
    entry = model_dump(backend)
    if hasattr(backend, "config"):
        entry["config"] = model_dump(backend.config)
    return entry
