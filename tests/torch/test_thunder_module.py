import numpy as np
import pytest
import torch
from lightning import Trainer
from lightning.pytorch.demos.boring_classes import RandomDataset
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from thunder import ThunderModule
from thunder.policy import Schedule


class BoringModel(ThunderModule):
    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64))


def loss(preds, labels=None):
    if labels is None:
        labels = torch.ones_like(preds)
    return torch.nn.functional.mse_loss(preds, labels)


@pytest.fixture
def architecture():
    return torch.nn.Linear(32, 2)


@pytest.fixture
def trainer(tmpdir):
    return Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=4,
        limit_test_batches=1,
        log_every_n_steps=1,
        logger=CSVLogger(tmpdir),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )


def test_configure_optimizers(architecture, trainer):
    sgd = SGD(architecture.parameters(), 1e-3)
    model = BoringModel(architecture, loss, -1, optimizer=sgd, lr_scheduler=Schedule(np.cos))
    trainer.fit(model)

    # test 1 optimizer and 2 schedulers
    model = BoringModel(architecture, loss, -1, optimizer=sgd, lr_scheduler=[Schedule(np.cos), Schedule(np.cos)])
    with pytest.raises(ValueError, match="got 1 and 2"):
        trainer.fit(model)

    # test 2 optimizers and 1 scheduler
    class BoringManyOptim(BoringModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.automatic_optimization = False

    sgds = [SGD(architecture.parameters(), 1e-3), SGD(architecture.parameters(), 1e-3)]
    model = BoringManyOptim(architecture, loss, -1, optimizer=sgds, lr_scheduler=Schedule(np.cos))
    trainer.fit(model)
    opt1, opt2 = model.optimizers()
    lr_sch = model.lr_schedulers()
    assert opt1.optimizer is lr_sch.optimizer

    # test with torch optimizer
    model = BoringManyOptim(architecture, loss, -1, optimizer=sgds, lr_scheduler=LambdaLR(sgd, lr_lambda=np.sin))
    trainer.fit(model)
    opt1, opt2 = model.optimizers()
    lr_sch = model.lr_schedulers()
    assert opt1.optimizer is lr_sch.optimizer

    # test 3 optimizers and 2 schedulers

    sgds = [SGD(architecture.parameters(), 1e-3) for _ in range(3)]
    schedulers = [Schedule(np.cos), LambdaLR(sgds[1], lr_lambda=np.exp)]
    model = BoringManyOptim(architecture, loss, -1, optimizer=sgds, lr_scheduler=schedulers)
    trainer.fit(model)

    opt1, opt2, opt3 = model.optimizers()
    lr_sch1, lr_sch2 = model.lr_schedulers()

    assert opt1.optimizer is lr_sch1.optimizer and opt2.optimizer is lr_sch2.optimizer

    # test no initialization
    with pytest.raises(NotImplementedError):
        model = BoringManyOptim(architecture, loss, -1)
        trainer.fit(model)


@pytest.fixture
def classifier():
    conv_block = lambda c_in, c_out: nn.Sequential(
        nn.Conv2d(c_in, c_out, 3), nn.BatchNorm2d(c_out), nn.ReLU()
    )
    return nn.Sequential(
        conv_block(1, 16),
        conv_block(16, 32),
        nn.MaxPool2d(2, 2),
        conv_block(32, 32),
        conv_block(32, 16),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 10),
    )
