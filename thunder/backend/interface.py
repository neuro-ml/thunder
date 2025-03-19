from pathlib import Path
from typing import Dict, Optional, Sequence, Type

from pydantic import BaseModel

from ..layout import Node
from ..pydantic_compat import PYDANTIC_MAJOR, NoExtra, field_validator, model_validate


class BackendConfig(NoExtra):
    """Backend Parameters"""


class Backend:
    Config: Type[BackendConfig]

    @staticmethod
    def run(config: BackendConfig, experiment: Path, nodes: Optional[Sequence[Node]], wait: Optional[bool] = None):
        """Start running the given `nodes` of an experiment located at the given path"""


class BackendEntryConfig(NoExtra):
    backend: str
    config: BackendConfig

    @field_validator("config", mode="before")
    def _val_config(cls, v, values):
        return parse_backend_config(v, values)

    @property
    def backend_cls(self):
        return backends[self.backend]


if PYDANTIC_MAJOR == 2:
    def parse_backend_config(v, values):
        val = backends[values.data["backend"]]
        return model_validate(val.Config, v)
else:
    def parse_backend_config(v, values):
        val = backends[values["backend"]]
        return model_validate(val.Config, v)


class MetaEntry(BaseModel):
    """
    Default backend set by `thunder backend set`
    """
    default: str


backends: Dict[str, Backend] = {}
