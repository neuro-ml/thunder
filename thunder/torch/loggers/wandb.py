import os
from pathlib import Path
from typing import Union

import wandb
from lightning.pytorch.loggers import WandbLogger as _WandbLogger


class WandbLogger(_WandbLogger):
    def __init__(self, name=None, save_dir='.', version=None,
                 offline=False, dir=None, id=None, anonymous=None, project=None,
                 log_model=False, experiment=None,
                 remove_dead_duplicates: bool = False, allow_rerun: bool = False, prefix='',
                 checkpoint_name=None, **kwargs):

        if allow_rerun:
            if project is None:
                raise ValueError("If you want WandB to load old logs, you should specify 'project'")
            versions = list(Path(save_dir).glob(f"{project}/*"))
            if versions:
                version = max(versions, key=lambda t: os.stat(t).st_mtime).name

        super().__init__(name, save_dir, version, offline, dir, id, anonymous, project,
                         log_model, experiment, prefix, checkpoint_name, **kwargs)

        if remove_dead_duplicates and not allow_rerun:
            api, exp = wandb.Api(), self.experiment
            for run in api.runs(path=f"{exp.entity}/{exp.project}"):
                if run.state in ["crashed", "failed"]:
                    if _same_group(run.group, exp.group) and run.name == exp.name:
                        run.delete()

    def __del__(self) -> None:
        wandb.finish()


def _same_group(run_group: Union[str, None], exp_group: str) -> bool:
    if run_group is None:
        return not exp_group
    return run_group == exp_group
