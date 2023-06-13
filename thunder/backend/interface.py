from pathlib import Path
from typing import Optional, Sequence, Type

from pydantic import BaseModel, Extra, validator

from ..layout import Node


class BackendConfig(BaseModel):
    class Config:
        extra = Extra.ignore


class Backend:
    Config: Type[BackendConfig]

    @staticmethod
    def run(config: BackendConfig, experiment: Path, nodes: Optional[Sequence[Node]]):
        pass


class BackendEntryConfig(BaseModel):
    backend: str
    config: BackendConfig

    @validator('config', pre=True)
    def _val_config(cls, v, values):
        val = backends[values['backend']]
        return val.Config.parse_obj(v)

    @property
    def backend_cls(self):
        return backends[self.backend]

    class Config:
        extra = Extra.ignore


backends = {}
