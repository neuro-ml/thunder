import random
from functools import wraps
from string import ascii_lowercase

import torch
from lightning import Trainer
from lightning.pytorch.demos.boring_classes import Net
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset as _Dataset

from thunder import ThunderModule
from thunder.callbacks import InferenceRunner


class Dataset(_Dataset):
    def __init__(self, ids):
        super().__init__()
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        return torch.randn(1, 28, 28), torch.tensor(random.randint(0, 9))


architecture = Net()

train_data = Dataset(list(range(10)))
val_data = Dataset(list(range(5)))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=3)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=2)
optimizer = Adam(architecture.parameters())

module = ThunderModule(architecture, nn.CrossEntropyLoss(), optimizer=optimizer)


def add_remove_dim(func):
    @wraps(func)
    def wrapper(x, *args, **kwargs):
        return func(x[None, ...], *args, **kwargs)[0]
    return wrapper


def test_no_additional_callbacks(tmpdir):
    inference_runner = InferenceRunner(predict_fn=add_remove_dim(module.predict),
                                       load_x=lambda i: val_data[i][0], load_y=lambda i: val_data[i][1],
                                       val_ids=list(ascii_lowercase), test_ids=list(ascii_lowercase.upper()))

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, callbacks=[inference_runner])

    trainer.fit(module, train_loader, val_loader)
    trainer.test(module, val_loader)


def test_with_additional_callback(tmpdir):
    """TODO"""


def test_with_empty_val_loader(tmpdir):
    """TODO"""
