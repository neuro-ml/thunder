import wandb
from lightning.pytorch.loggers import WandbLogger as _WandbLogger


class WandbLogger(_WandbLogger):
    def __init__(self, name=None, save_dir='.', version=None,
                 offline=False, dir=None, id=None, anonymous=None, project=None,
                 log_model=False, experiment=None,
                 remove_dead_duplicates: bool = False, prefix='', checkpoint_name=None, **kwargs):
        super().__init__(name, save_dir, version, offline, dir, id, anonymous, project,
                         log_model, experiment, prefix, checkpoint_name, **kwargs)

        if remove_dead_duplicates:
            api = wandb.Api()
            for run in api.runs(path=f"{self.experiment.entity}/{self.experiment.project}"):
                if run.state != "running":
                    if run.group == self.experiment.group or run.name == self.experiment.name:
                        run.delete()
