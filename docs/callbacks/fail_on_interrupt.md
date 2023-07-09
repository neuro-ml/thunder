# FailOnInterrupt
Forces lightning Trainer to fail on KeyboardInterrupt by raising RuntimeError.

## Usage
```python
from lightning import Trainer
from thunder.callbacks import FailOnInterrupt

trainer = Trainer(..., callbacks=[FailOnInterrupt()])
```

## Reference
::: thunder.callbacks.fail_on_interrupt.FailOnInterrupt
    handler: python
    options:
      members:
        - on_exception
      show_root_heading: true
      show_source: true
