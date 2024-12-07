```python
import numpy as np
import torch
from lightning import Trainer
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import accuracy_score
from thunder import ThunderModule
from thunder.callbacks import MetricMonitor
from thunder.placeholders import ExpName, GroupName
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


# will be executed just before `thunder run`
CALLBACKS = [seed_everything()]

BATCH_SIZE = 256

train_ds = MNIST(".", train=True, download=True, transform=transforms.ToTensor())
val_ds = MNIST(".", train=False, download=True, transform=transforms.ToTensor())
train_data = DataLoader(train_ds, batch_size=BATCH_SIZE)
val_data = DataLoader(val_ds, batch_size=BATCH_SIZE)

architecture = nn.Sequential(nn.Flatten(), torch.nn.Linear(28 * 28, 10))

module = ThunderModule(
    architecture, nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(architecture.parameters())
)

# Preparing metrics
# 'y' and 'x' are single label and 
# model prediction for a single image,
# hence the 'np.argmax(x)' for extracting
# the predicted label.
group_accuracy = {lambda y, x: (y, np.argmax(x)): accuracy_score}

# Initialize a trainer
trainer = Trainer(
    callbacks=[ModelCheckpoint(save_last=True),
               MetricMonitor(group_metrics=group_accuracy)],
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
## Source
Full source code is available at [thunder-examples](https://github.com/arseniybelkov/thunder-examples/blob/master/configs/mnist.config)