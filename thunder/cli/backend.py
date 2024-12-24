import copy
import functools
from collections import ChainMap
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import typer
import yaml
from click import Context
from typer import Option
from typer.core import TyperCommand
from typer.main import get_click_param
from typer.models import ParamMeta

from ..backend import BackendEntryConfig, MetaEntry, backends
from ..pydantic_compat import model_validate, resolve_pydantic_major
from .app import app


BACKENDS_CONFIG_PATH = Path(typer.get_app_dir(app.info.name)) / 'backends.yml'


class BackendCommand(TyperCommand):
    _current_backend: Optional[str] = None

    @staticmethod
    def get_backend(backend: str, kwargs: dict):
        kwargs = kwargs.copy()
        duplicate = kwargs.pop('kwargs', None)
        assert duplicate is None, duplicate
        backend = collect_backends()[backend]
        return backend, backend.Config(**kwargs)

    def parse_args(self, ctx: Context, args):
        self._current_backend = None
        try:
            index = args.index('--backend')
        except ValueError:
            pass
        else:
            if index < len(args) - 1:
                self._current_backend = args[index + 1]

        return super(BackendCommand, self).parse_args(ctx, args)

    @property
    def params(self):
        # TODO: check that the last 2 args are backend and kwargs
        return self._params[:-2] + [get_click_param(x)[0] for x in populate(self._current_backend)]

    @params.setter
    def params(self, value):
        self._params = value


def populate(backend_name):
    configs, meta = collect_configs()
    local_configs = sorted(set(configs) - set(backends))
    if backend_name is None:
        if meta is not None:
            entry = configs[meta.default]
        elif len(local_configs) == 1:
            entry = configs[local_configs[0]]
        elif len(local_configs) > 1:
            entry = None
        else:
            entry = configs["cli"]

    else:
        if backend_name in configs:
            entry = configs[backend_name]
        else:
            raise ValueError(f"Specified backend `{backend_name} is not among "
                             f"available configs: {sorted(configs)}`")

    backend_choices = ", ".join(sorted(configs)).rstrip(", ")
    if entry is None:
        return [ParamMeta(
            name='backend', annotation=Optional[str],
            default=Option(
                ..., help=f'The runner backend to use. Choices: {backend_choices}.',
                show_default=False,
            )
        )]

    backend_name = entry.backend
    backend_params = [ParamMeta(
        name='backend', annotation=Optional[str],
        default=Option(
            backend_name,
            help=f'The runner backend to use. Choices: {backend_choices}. Currently using {backend_name}. '
                 f'List of backends can be found at {str(BACKENDS_CONFIG_PATH.resolve())}',
            show_default=False,
        ),
    )]
    backend_params.extend(_collect_backend_params(entry, backend_name))
    return backend_params


if resolve_pydantic_major() >= 2:
    def _collect_backend_params(entry, backend_name):
        """
        Config Annotation depends on pydantic version.
        """
        for field_name, field in entry.backend_cls.Config.model_fields.items():
            field_clone = copy.deepcopy(field)
            field_clone.default = getattr(entry.config, field_name)
            yield ParamMeta(
                name=field_name, default=field_clone.default, annotation=field.annotation,
            )
else:
    def _collect_backend_params(entry, backend_name):
        for field in entry.backend_cls.Config.__fields__.values():
            annotation = field.outer_type_
            # TODO: https://stackoverflow.com/a/68337036
            if not hasattr(annotation, '__metadata__') or not hasattr(annotation, '__origin__'):
                raise ValueError('Please use the `Annotated` syntax to annotate you backend config')

            # TODO
            default, = annotation.__metadata__
            default = copy.deepcopy(default)
            default.default = getattr(entry.config, field.name)
            default.help = f'[{backend_name} backend] {default.help}'
            yield ParamMeta(
                name=field.name, default=default, annotation=annotation.__origin__,
            )


def collect_backends() -> ChainMap:
    """
    Collects backend for each config.
    Returns
    -------
    ChainMap[str, Backend]
    mapping config_name : backend
    """
    configs, _ = collect_configs()
    local_backends = {name: backends[config.backend] for name, config in configs.items()}
    return ChainMap(backends, local_backends)


@functools.lru_cache()
def collect_configs() -> Tuple[ChainMap, Union[MetaEntry, None]]:
    """
    Collects configs for `thunder run` command.
    Returns
    -------
    (mapping, meta) : Tuple[ChainMap[str, BackendEntryConfig], Union[MetaEntry, None]]
    mapping - mapping config_name : BackendEntryConfig
    meta - meta info (e.g. default backend), if no meta data found, returns None
    """
    local_configs = load_backend_configs()
    builtin_configs = {
        name: BackendEntryConfig(backend=name, config={})
        for name in backends.keys()
    }
    meta = local_configs.pop("meta", None)
    return ChainMap(builtin_configs, local_configs), meta


def load_backend_configs() -> Dict[str, Union[BackendEntryConfig, MetaEntry]]:
    path = BACKENDS_CONFIG_PATH
    if not path.exists():
        # TODO: return Option[Dict]
        return {}

    with path.open('r') as file:
        local = yaml.safe_load(file)
    if local is None:
        return {}
    # FIXME
    assert isinstance(local, dict), type(local)
    return {k: model_validate(BackendEntryConfig, v)
            if k != "meta" else model_validate(MetaEntry, v) for k, v in local.items()}
