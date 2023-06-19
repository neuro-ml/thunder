from itertools import combinations

import numpy as np
import pytest
import torch
from lightning import Trainer
from lightning.pytorch.demos.boring_classes import RandomDataset
from lightning.pytorch.loggers import CSVLogger
from more_itertools import zip_equal
from torch.optim import SGD
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


class BoringManyOptim(BoringModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False


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


def test_1_optimizer_2_schedulers(architecture, trainer):
    sgd = SGD(architecture.parameters(), 1e-3)
    model = BoringModel(architecture, loss, -1, optimizer=sgd, lr_scheduler=[Schedule(np.cos), Schedule(np.cos)])
    with pytest.raises(ValueError, match="received None"):
        trainer.fit(model)


def test_2_optimizers_2_schedulers(architecture, trainer):
    sgds = [SGD(architecture.parameters(), 1e-3), SGD(architecture.parameters(), 1e-3)]
    model = BoringManyOptim(architecture, loss, -1, optimizer=sgds, lr_scheduler=Schedule(np.cos))
    trainer.fit(model)
    opt1, opt2 = model.optimizers()
    lr_sch = model.lr_schedulers()
    assert all(map(lambda o: o.optimizer is not None, model.optimizers()))
    assert opt1.optimizer is lr_sch.optimizer


def test_with_torch_optimizer(architecture, trainer):
    sgds = [SGD(architecture.parameters(), 1e-3), SGD(architecture.parameters(), 1e-3)]
    # TODO sgds[1] misalignment
    model = BoringManyOptim(architecture, loss, -1, optimizer=sgds, lr_scheduler=LambdaLR(sgds[0], lr_lambda=np.sin))
    trainer.fit(model)
    opt1, opt2 = model.optimizers()
    lr_sch = model.lr_schedulers()
    check_optimizers(model.optimizers())
    assert opt1.optimizer is lr_sch.optimizer


# def test_torch_schedulers_misalignment(architecture, trainer):
#     sgds = [SGD(architecture.parameters(), 1e-3), SGD(architecture.parameters(), 1e-3)]
#     # TODO sgds[1] misalignment
#     model = BoringManyOptim(architecture, loss, -1, optimizer=sgds, lr_scheduler=LambdaLR(sgds[1], lr_lambda=np.sin))
#     trainer.fit(model)
#     opt1, opt2 = model.optimizers()
#     lr_sch = model.lr_schedulers()
#     check_optimizers(model.optimizers())
#     assert opt1.optimizer is lr_sch.optimizer


def test_3_optimizers_2_schedulers(architecture, trainer):
    sgds = [SGD(architecture.parameters(), 1e-3) for _ in range(3)]
    schedulers = [Schedule(np.cos), LambdaLR(sgds[1], lr_lambda=np.exp)]
    model = BoringManyOptim(architecture, loss, -1, optimizer=sgds, lr_scheduler=schedulers)
    trainer.fit(model)

    opt1, opt2, opt3 = model.optimizers()
    lr_sch1, lr_sch2 = model.lr_schedulers()

    assert all(map(lambda o: o.optimizer is not None, model.optimizers()))
    assert opt1.optimizer is lr_sch1.optimizer and opt2.optimizer is lr_sch2.optimizer


def test_no_initialization(architecture, trainer):
    with pytest.raises(NotImplementedError):
        model = BoringManyOptim(architecture, loss, -1)
        trainer.fit(model)


def test_no_schedulers(architecture, trainer):
    sgds = [SGD(architecture.parameters(), 1e-3) for _ in range(3)]
    model = BoringManyOptim(architecture, loss, -1, optimizer=sgds)
    trainer.fit(model)

    opt1, opt2, opt3 = model.optimizers()
    assert model.lr_schedulers() is None
    assert all(map(lambda o: o.optimizer is not None, model.optimizers()))


def test_no_optimizers_and_only_torch_schedulers(architecture, trainer):
    sgds = [SGD(architecture.parameters(), 1e-3), SGD(architecture.parameters(), 1e-3)]
    schedulers = [LambdaLR(sgds[0], lr_lambda=np.exp),
                  LambdaLR(sgds[1], lr_lambda=np.cos)]
    model = BoringManyOptim(architecture, loss, -1, lr_scheduler=schedulers)
    trainer.fit(model)

    for sch, opt in zip_equal(model.lr_schedulers(), model.optimizers()):
        assert sch.optimizer is opt.optimizer
        assert opt.optimizer is not None


def check_optimizers(optimizers):
    assert all(map(lambda o: o.optimizer is not None, optimizers))
    assert all(map(lambda oo: oo[0].optimizer is not oo[1].optimizer, combinations(optimizers, 2)))
