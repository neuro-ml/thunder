from typing import Callable, Sequence, Tuple, Union

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, TQDMProgressBar
from lightning.pytorch.trainer.call import _call_callback_hooks, _call_lightning_module_hook
from more_itertools import zip_equal
from toolz import compose

from ..torch.utils import maybe_from_np


Loader = Tuple[Sequence, Callable, Callable]


class InferenceRunner(Callback):
    def __init__(
        self,
        *decorators: Callable,
        val_loaders: Union[Loader, Sequence[Loader]] = None,
        test_loaders: Union[Loader, Sequence[Loader]] = None,
        predict_loaders: Union[Loader, Sequence[Loader]] = None,
    ):
        """
        Run inference on different stages, allowing you to get individual metrics.
        Parameters
        ----------
        *decorators : Callable
            Decorators applied to pl_module.
        val_loaders : Union[Loader, Sequence[Loader]]
        test_loaders : Union[Loader, Sequence[Loader]]
        predict_loaders : Union[Loader, Sequence[Loader]]
        """

        def _wrap_loader(loader):
            if len(loader) == 3:
                return [loader] if callable(loader[1]) else loader
            return loader

        self.decorators = compose(*decorators)
        self.val_loaders = _wrap_loader(val_loaders or [])
        self.test_loaders = _wrap_loader(test_loaders or [])
        self.predict_loaders = _wrap_loader(predict_loaders or [])

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        @self.decorators
        def predict(x):
            if not isinstance(x, tuple):
                x = (x,)
            return pl_module(*maybe_from_np(x, device=pl_module.device))

        self._predict = predict
        trainer.callbacks = [c for c in trainer.callbacks if not isinstance(c, TQDMProgressBar)]

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        delattr(self, "_predict")

    def evaluate_epoch(self, trainer: Trainer, stage: str, loaders: Sequence[Loader]) -> None:
        for dataloader_idx, (ids, load_x, load_y) in enumerate(loaders):
            for idx, x, y in zip_equal(ids, map(load_x, ids), map(load_y, ids)):
                args = ((x[None, ...], y[None, ...]), idx, dataloader_idx)

                hook_name = f"on_{stage}_batch_start"
                _call_callback_hooks(trainer, hook_name, *args)
                _call_lightning_module_hook(trainer, hook_name, *args)

                predict = self._predict(x)

                hook_name = f"on_{stage}_batch_end"
                _call_callback_hooks(trainer, hook_name, (predict[None, ...], y[None, ...]), *args)
                _call_lightning_module_hook(trainer, hook_name, (predict[None, ...], y[None, ...]), *args)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.evaluate_epoch(trainer, "validation", self.val_loaders)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.evaluate_epoch(trainer, "test", self.test_loaders)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.evaluate_epoch(trainer, "predict", self.predict_loaders)
