import numpy as np
import pytest
import torch
from lightning import Trainer
from lightning.pytorch.demos.boring_classes import RandomDataset
from lightning.pytorch.loggers import CSVLogger
from torch.optim import SGD
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
    model = BoringModel(architecture, loss, -1, sgd, Schedule(np.cos))
    trainer.fit(model)

    # test 1 optimizer and 2 schedulers
    model = BoringModel(architecture, loss, -1, sgd, [Schedule(np.cos), Schedule(np.cos)])
    with pytest.raises(ValueError, match="got 1 and 2"):
        trainer.fit(model)

    # test 2 optimizers and 1 scheduler
    class BoringManyOptim(BoringModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            o1, o2 = self.optimizers()
            sch = self.lr_schedulers()

            o1.zero_grad(), o2.zero_grad()
            self.manual_backward(loss)
            o1.step()

    sgd = SGD(architecture.parameters(), 1e-3)
    model = BoringManyOptim(architecture, loss, -1, sgd, Schedule(np.cos))
    trainer.fit(model)

    # test 3 optimizers and 2 schedulers

    sgds = [SGD(architecture.parameters(), 1e-3) for _ in range(3)]
    schedulers = [Schedule(np.cos), Schedule(np.exp), Schedule(np.sin)]
    model = BoringManyOptim(architecture, loss, -1, sgds, schedulers)
    trainer.fit(model)