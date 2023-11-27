# SeedEverything
Executes `lightning.seed_everything` when being initialized.

## Usage
```python
from lightning import Trainer
from thunder.callbacks import SeedEverything

trainer = Trainer(..., callbacks=[SeedEverything(0xBadCafe)])
## INFO: Global seed set to 0xBadCafe
```

## Reference
::: thunder.callbacks.seed_everything.SeedEverything
    handler: python
    options:
      members:
        - on_exception
      show_root_heading: true
      show_source: true
