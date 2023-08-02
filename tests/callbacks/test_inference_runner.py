import random
from functools import wraps
from string import ascii_lowercase

import torch
from lightning import Trainer
from lightning.pytorch.demos.boring_classes import Net
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset as _Dataset

from thunder import ThunderModule
from thunder.callbacks import InferenceRunner, TimeProfiler


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


def load_x(i):
    return val_data[i][0]


def load_y(i):
    return val_data[i][1]


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
    inference_runner = InferenceRunner(add_remove_dim, val_loaders=(ascii_lowercase, load_x, load_y))

    trainer = Trainer(default_root_dir=tmpdir,
                      max_epochs=2,
                      callbacks=[inference_runner],
                      enable_checkpointing=False,
                      enable_progress_bar=False)

    trainer.fit(module, train_loader, val_loader)
    trainer.test(module, val_loader)


def test_with_additional_callback(tmpdir):
    inference_runner = InferenceRunner(add_remove_dim, val_loaders=(ascii_lowercase, load_x, load_y),
                                       test_loaders=(ascii_lowercase.upper(), load_x, load_y))

    trainer = Trainer(default_root_dir=tmpdir,
                      max_epochs=2,
                      callbacks=[inference_runner, TimeProfiler()],
                      enable_checkpointing=False,
                      enable_progress_bar=False)

    trainer.fit(module, train_loader, val_loader)
    trainer.test(module, val_loader)


def test_with_empty_val_loader(tmpdir):
    inference_runner = InferenceRunner(add_remove_dim, val_loaders=(ascii_lowercase, load_x, load_y),
                                       test_loaders=(ascii_lowercase.upper(), load_x, load_y),
                                       predict_loaders=[(list(range(len(ascii_lowercase))), load_x, load_y)])

    trainer = Trainer(default_root_dir=tmpdir,
                      max_epochs=2,
                      callbacks=[inference_runner],
                      enable_checkpointing=False,
                      enable_progress_bar=False)

    trainer.fit(module, train_loader, [])
    trainer.test(module, val_loader)
    trainer.predict(module, [])
