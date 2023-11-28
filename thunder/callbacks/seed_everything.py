from typing import Any, Dict

from lightning import Callback, seed_everything


class SeedEverything(Callback):
    def __init__(self, seed: int = 42, workers: bool = False):
        self.seed = seed
        self.workers = workers
        seed_everything(seed, workers)

    def state_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "workers": self.workers}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.seed = state_dict["seed"]
        self.workers = state_dict["workers"]
