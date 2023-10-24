# ThunderModule

ThunderModule inherits everything from LightningModule and implements
essential methods for most common training pipelines.

## From Lightning to Thunder
Most common pipelines are implemented in lightning in the following way:
```python
from lightning import LightningModule

class Model(LightningModule):
    def __init__(self):
        self.architecture: nn.Module = ...
        self.metrics = ... # smth like Dict[str, Callable]
        
    def forward(self, *args, **kwargs):
        return self.architecture(*args, **kwargs)
    
    def criterion(self, x, y):
        ...
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.criterion(self(x), y)
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        # forward and metrics computation or output preservation
        ...
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        # forward and metrics computation or output preservation
        ...
    
    def configure_optimizers(self):
        return Adam(...), StepLR(...)
```

ThunderModule offers an implementation of necessary steps shown above.
```python
from thunder import ThunderModule

architecture: nn.Module = ...
criterion = CrossEntropy()
optimizer = Adam(architecture.parameters())
scheduler = StepLR(optimizer)

model = ThunderModule(architecture, criterion,
                      optimizer=optimizer, lr_scheduler=scheduler)
```

## Configuring Optimizers
For extra information see
[this](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#lightningmodule-api).  
Lightning requires optimizers and learning rate policies
to be defined inside `configure_optimizers` method.  
Using ThunderModule allows you to pass the following configurations of 
optimizers and learning rate schedulers:

```python
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Adam

architecture = nn.Linear(2, 2)
```
#### No scheduling
```python
optimizer = Adam(architecture.parameters())
model = ThunderModule(..., optimizer=optimizer)
```
#### Defining optimizer and scheduler
```python
optimizer = Adam(architecture.parameters())
lr_scheduler = LRScheduler(optimizer)
model = ThunderModule(..., optimizer=optimizer, lr_scheduler=lr_scheduler)
```
#### Defining no optimizer
```python
lr_scheduler = LRScheduler(optimizer)
model = ThunderModule(..., lr_scheduler=lr_scheduler)
```

### Multiple Optimizers
Thunder just as lightning supports configuration with more than 1 optimizer. If such configuration is to be used, manual optimization is required.  
[Guide on manual optimization](https://lightning.ai/docs/pytorch/stable/common/optimization.html#id2)

In thunder you can pass lists of optimizers and schedulers to ThunderModule.
```python
class ThunderModuleManual(ThunderModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False


optimizers = [Adam(module1.parameters()), Adam(module2.parameters())]
lr_schedulers = [Scheduler(opt) for opt in optimizers]

model = ThunderModuleManual(..., optimizer=optimizers, lr_scheduler=lr_schedulers)
```

### Thunder Policies
As shown above, torch schedulers require optimizer(s) to be passed to them before
they are given to ThunderModule. It is not very convenient, and also they lack some basic 
functionality.  
You can use thunder policies just like torch schedulers:
```python
from thunder.policy import Switch

optimizers = [Adam(module1.parameters()), Adam(module2.parameters())]
lr_schedulers = [Switch({1: 0.001}), Switch({2: 0.001})]

model = ThunderModuleManual(..., optimizer=optimizers, lr_scheduler=lr_schedulers)
```

For extra information see [Thunder Policies Docs](../policy/lr_schedulers.md).

## Inference
During inference step, ThunderModule uses Predictors in order to preprocess data and
make inverse transforms after passing data through the model. Default predictor
is just an identity function.

For more on predictors see [Thunder Predictors Docs](../inference/index.md).

## Batch Transfer
ThunderModule transfers training batches to device by default. However, during 
inference batch remains on the device, on which it was received from data loader. 
Transferring happens later in the `inference_step`, which is invoked in
`validation_step`, `test_step` and `predict_step`.

## Reference
::: thunder.torch.core.ThunderModule
    handler: python
    options:
      members:
        - init
        - training_step
        - validation_step
        - test_step
        - predict_step
        - inference_step
      show_root_heading: true
      show_source: true
