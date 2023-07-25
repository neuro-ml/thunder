import logging
from typing import List, Any, Optional, Callable, Sequence

from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback, Checkpoint, EarlyStopping

log = logging.getLogger(__name__)


class InferenceRunner(Callback):
    def __init__(
        self,
        *,
        predict_fn: Callable,
        load_x: Callable,
        load_y: Callable,
        test_load_x: Optional[Callable] = None,
        test_load_y: Optional[Callable] = None,
        val_ids: Optional[Sequence] = None,
        test_ids: Optional[Sequence] = None,
        callbacks: List[Callback] = None,
    ):
        self.predict_fn = predict_fn
        self.load_x = load_x
        self.load_y = load_y
        self.test_load_x = test_load_x or load_x
        self.test_load_y = test_load_y or load_y
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.callbacks = callbacks if callbacks else []

    def evaluate_epoch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
        ids: Sequence,
        load_x: Callable,
        load_y: Callable,
    ) -> None:
        for i, x, y in zip(ids, map(load_x, ids), map(load_y, ids)):
            self._call_callback_hooks(trainer, pl_module, f"on_{stage}_batch_start",
                                      batch=(x, y), batch_idx=i, dataloader_id=0)

            predict = self.predict_fn(x)

            self._call_callback_hooks(trainer, pl_module, f"on_{stage}_batch_end",
                                      outputs=(predict, y), batch=(x, y), batch_idx=i, dataloader_id=0)

        self._call_callback_hooks(trainer, pl_module, f"on_{stage}_epoch_end")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.evaluate_epoch(trainer, pl_module, "validation", self.val_ids, self.load_x, self.load_y)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.evaluate_epoch(trainer, pl_module, "test", self.test_ids, self.test_load_x, self.test_load_y)

    def _call_callback_hooks(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        hook_name: str,
        *args: Any,
        monitoring_callbacks: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        log.debug(f"{self.__class__.__name__}: calling callback hook: {hook_name}")

        if pl_module:
            prev_fx_name = pl_module._current_fx_name
            pl_module._current_fx_name = hook_name

        callbacks = self.callbacks
        if monitoring_callbacks is True:
            # the list of "monitoring callbacks" is hard-coded to these two. we could add an API to define this
            callbacks = [cb for cb in callbacks if isinstance(cb, (EarlyStopping, Checkpoint))]
        elif monitoring_callbacks is False:
            callbacks = [cb for cb in callbacks if not isinstance(cb, (EarlyStopping, Checkpoint))]

        for callback in callbacks:
            fn = getattr(callback, hook_name)
            if callable(fn):
                with trainer.profiler.profile(f"[Callback]{callback.state_key}.{hook_name}"):
                    fn(trainer, pl_module, *args, **kwargs)

        if pl_module:
            # restore current_fx when nested context
            pl_module._current_fx_name = prev_fx_name
