import inspect
import sys
from inspect import Parameter
from pathlib import Path
from typing import Optional

from typing_extensions import Annotated
import typer
import yaml
from typer import Option

from .main import app, run, build_run
from .wandb import wand_app, agent
from ..backend import Cli, BackendEntryConfig


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

    for cmd in run, build_run:
        patch_backend(app, cmd, backend, backend_name, backend_params)
    patch_backend(wand_app, agent, backend, backend_name, backend_params)

    app.add_typer(wand_app)
    app()


def patch_backend(typer_app, cmd, backend, backend_name, backend_params):
    def func(**kwargs):
        assert kwargs.pop('backend') == backend_name, kwargs.pop('backend')
        return cmd(**kwargs, backend=backend)

    common_params = list(inspect.signature(cmd).parameters.values())[:-2]
    func.__signature__ = inspect.Signature(common_params + backend_params)
    func.__name__ = cmd.__name__
    func.__doc__ = cmd.__doc__
    typer_app.command()(func)


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
