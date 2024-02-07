# WandbLogger
Slightly modified [WandbLogger](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb).  
`thunder` version has additional parameter `remove_dead_duplicates` and `allow_rerun` 
that if being set to `True` (`False` by default), deletes all crashed or failed runs 
with the same name and group within your project.
```python
from thunder.torch.loggers import WandbLogger
logger = WandbLogger(..., remove_dead_duplicates=True)
```

`allow_rerun` equals `True` disables `remove_dead_duplicates` and seeks  
last experiment WandB has created in your experiment folder. 
If your experiment failed during the previous run and you restarted it with `allow_rerun = True`.
WandB will not create new versions, but will use the last run for logging and checkpointing.
