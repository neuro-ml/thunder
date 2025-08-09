from pathlib import Path
from typing import Dict, Optional, Sequence, Type

from pydantic import BaseModel, field_validator

from ..layout import Node


class NoExtra(BaseModel):
    model_config = {
        "extra": "forbid",
    }


class EngineConfig(NoExtra):
    """Backend Parameters"""


class Engine:
    Config: Type[EngineConfig]

    @staticmethod
    def run(config: EngineConfig, experiment: Path, nodes: Optional[Sequence[Node]], wait: Optional[bool] = None):
        """Start running the given `nodes` of an experiment located at the given path"""


class BackendEntryConfig(NoExtra):
    backend: str
    config: EngineConfig

    @field_validator("config", mode="before")
    def _val_config(cls, v, values):
        return parse_backend_config(v, values)

    @property
    def backend_cls(self):
        return engines[self.backend]


def parse_backend_config(v, values):
    val = engines[values.data["backend"]]
    return val.Config.model_validate(v)


class MetaEntry(BaseModel):
    """
    Default backend set by `thunder backend set`
    """

    default: str


engines: Dict[str, Engine] = {}
