# Low Resolution Liver Segmentation
##### Requirements: [deep-pipe](https://github.com/neuro-ml/deep_pipe), [amid](https://github.com/neuro-ml/amid), [connectome](https://github.com/neuro-ml/connectome)
Deep-Pipe was primarly used for metrics and batch combinations.

## Main config

```python
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from amid.totalsegmentator import Totalsegmentator
from connectome import Apply, CacheToDisk, CacheToRam, Chain, Filter
from dpipe import layers
from dpipe.batch_iter import combine_pad
from dpipe.im.metrics import dice_score, precision, recall
from dpipe.torch.functional import weighted_cross_entropy_with_logits
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from thunder import ThunderModule
from thunder.callbacks import MetricMonitor, TimeProfiler
from thunder.layout import SingleSplit
from thunder.placeholders import ExpName, GroupName
from thunder.policy import Switch
from torch.optim import Adam
from torch.utils.data import DataLoader

from thunder_examples.dataset import (ConToTorch, NormalizeCT, RotateTotalsegm,
                                      Zoom)

SEED = 0xBadCafe

totalsegmentator = Totalsegmentator("/shared/data/Totalsegmentator_dataset.zip")
                   >> Filter(lambda study_type, split: study_type == "ct abdomen-pelvis" and split == "train")
preprocessing = Chain(RotateTotalsegm(), Zoom(n=0.3), NormalizeCT(max_=200, min_=-200))

dataset = Chain(totalsegmentator,
                preprocessing,
                CacheToRam())

layout = SingleSplit(dataset, train=0.7, val=0.3)

batch_size = 2
batches_per_epoch = 256
max_epochs = 200

train_data = DataLoader(
    ConToTorch(layout.train >> Apply(image=lambda x: x[None], liver=lambda x: x[None]), ['image', 'liver']),
    batch_size=batch_size, num_workers=4,
    shuffle=True, collate_fn=partial(combine_pad, padding_values=np.min))

val_data = DataLoader(
    ConToTorch(layout.val >> Apply(image=lambda x: x[None], liver=lambda x: x[None]), ['image', 'liver']),
    batch_size=batch_size, collate_fn=partial(combine_pad, padding_values=np.min), num_workers=4)

architecture = nn.Sequential(
    nn.Conv3d(1, 8, kernel_size=3, padding=1),

    layers.FPN(
        layers.ResBlock3d, nn.MaxPool3d(2), nn.Identity(),
        layers.fpn.interpolate_merge(lambda x, y: torch.cat([x, y], 1), order=1),
        [
            [[8, 16, 16], [32, 16, 8]],
            [[16, 32, 32], [64, 32, 16]],
            [[32, 64, 64], [128, 64, 32]],
            [[64, 128, 128], [256, 128, 64]],
            [128, 256, 128],
        ],
        kernel_size=3, padding=1,
    ),

    layers.PreActivation3d(8, 1, kernel_size=3, padding=1),
)

criterion = weighted_cross_entropy_with_logits

module = ThunderModule(architecture, criterion, activation=nn.Sigmoid(),
                       optimizer=Adam(architecture.parameters()),
                       lr_scheduler=Switch({0: 1e-3, 50: 1e-4, 150: 1e-5}))

trainer = Trainer(
    callbacks=[
        MetricMonitor({lambda y, x: (y > 0.5, x > 0.5): [precision, recall, dice_score]},
                      aggregate_fn=["std", "max", "min"]),
        TimeProfiler(),
        LearningRateMonitor("epoch"),
        ModelCheckpoint(save_last=True),
    ],
    limit_train_batches=batches_per_epoch,
    accelerator='gpu', precision=16,
    max_epochs=max_epochs,
    logger=WandbLogger(name=ExpName, group=GroupName, project='thunder-examples', entity='arseniybelkov'))
```
## ConToTorch
__ConTotch__ is a wrapper for connectome dataset for it can be passed to torch DataLoader.
```python
from torch.utils.data import Dataset


class ConToTorch(Dataset):
    def __init__(self, dataset, fields):
        self.loader = dataset._compile(fields)
        self.ids = dataset.ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        return self.loader(self.ids[item])
```

## Source
Full source code is available at [thunder-examples](https://github.com/arseniybelkov/thunder-examples/blob/master/configs/dumb.config)