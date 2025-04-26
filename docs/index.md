> _You saw the lightning. Now it's time to hear the thunder_ 🌩️

# Thunder

🌩️ The Deep Learning framework based on [Lightning](https://lightning.ai/)

## Install

```bash
pip install thunder
``` 

## Start experimenting
For the sake of simplicity (and sanity) we will get by with MNIST classification problem.

We need:
- Datasets & Dataloaders (train, validation)
- Model + optimizers & schedulers
- Training loop

### Writing your first configs
For running an experiment with 🌩️ you first need a `config`.
Only [lazycon](https://github.com/maxme1/lazycon) configs are currently supported.
Create `data.config` (it can be `anyname.config`), and just write usual python code 
(many of the non-thunder imports are intentionally omitted for simplicity).
```python
from torchvision.datasets import MNIST

BATCH_SIZE = 256 

train_ds = MNIST(".", train=True, download=True, transform=transforms.ToTensor())
val_ds = MNIST(".", train=False, download=True, transform=transforms.ToTensor())

train_data = DataLoader(train_ds, batch_size=BATCH_SIZE) # req
val_data = DataLoader(val_ds, batch_size=BATCH_SIZE) # req
```

Datasets are ready to deploy, now we need a model. Let's create `architecture.config`
(again, the name can be arbitrary). 
```python
from thunder import ThunderModule

architecture = nn.Sequential(nn.Flatten(), torch.nn.Linear(28 * 28, 10))

module = ThunderModule( # req
    architecture, nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(architecture.parameters())
)
```
In came [ThunderModule](./core/thunder_module). Basically, it's just 
a wrapper around `LightningModule` with implemented train, val and test steps. 
The implementation should suffice many of the popular Deep Learning tasks.

The experiment is almost ready to be assembled, yet it usually is nice
to have some metrics being calculated during the run.
The more the configs, the merrier. `metrics.config`:
```python
import numpy as np
from sklearn.metrics import accuracy_score
from thunder.callbacks import MetricMonitor

metric_monitor = MetricMonitor(group_metrics={lambda y, x: (np.argmax(y), x): accuracy_score})
```
[MetricMonitor](./callbacks/metric_monitor) is a 🌩️ callback, that keeps track
of losses and metrics. It has many options, check the docs for more insights.

And now, all we need is `Trainer`. With it we will assemble the config.  
`core.config`:  
```python
from .metrics import *
from .data import *
from .model import *
from lightning import Trainer

trainer = Trainer(
    callbacks=[ModelCheckpoint(save_last=True),
               metric_monitor],
    accelerator="auto",
    devices=1,
    max_epochs=100,
    logger=WandbLogger(
        name=ExpName,
        group=GroupName,
        project="thunder-examples",
        entity="arseniybelkov",
    ),
)
```
One can see that we import all the previously created config into one. 

### Building the experiment
Run
```commandline
thunder build /path/to/core.config /path/to/experiment_name
```
It will create a folder with build `experiment.config`. If you open it, you'll see 
that it is just merged `.config` files you imported.
### Running the experiment
```commandline
thunder run /path/to/experiment_name
```
The command above will run `trainer.fit` with dataloaders fetched from the 
built config. `build` and `run` commands have many additional parameters, you can read about them in the [documentation](./cli). 

## Core Features
- ### [ThunderModule](./core/thunder_module)
- ### [MetricMonitor](./callbacks/metric_monitor)
- ### [Experiment configs](./configs)
- ### [CLI & Integrations with WandB](./cli)
