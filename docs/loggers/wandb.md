# WandbLogger
Slightly modified [WandbLogger](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb).  
`thunder` version has additional parameter `remove_dead_duplicates` 
that if being set to `True` (`False` by default), deletes all crashed or failed runs 
with the same name and group within your project.
```python
from thunder.torch.loggers import WandbLogger
logger = WandbLogger(..., remove_dead_duplicates=True)
```