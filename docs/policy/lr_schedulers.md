# LR Schedulers

All schedulers in thunder are subclasses of `torch.optim.lr_scheduler.LRScheduler`.
However, during initialization they do not require optimizer to be passed.

## Usage
We will use Switch as an example.  
```python
from thunder.policy import Switch

switch = Switch({10: 0.001, 20: 0.001 / 10})
```
We have just created a policy, but to make it work, it still needs an optimizers.  
Let's see how it works after being assembled.  
```python
optimizer = Adam(...)
scheduler(optimizer) # binds optimizer to scheduler
# or 
# scheduler = scheduler(optimizer)
# You can also retrieve optimizer:
opt = scheduler.optimizer
```
After assigning optimizer to scheduler, policy instance will work just like usual
torch scheduler.

### Initial LR
All schedulers have `lr_init` parameters, if specified, it will be used as lr value on
0th step.

## Reference
::: thunder.policy.Multiply
    handler: python
    options:
      show_root_heading: true
      show_source: true
      show_base: true

::: thunder.policy.Schedule
    handler: python
    options:
      show_root_heading: true
      show_source: true
      show_base: true

::: thunder.policy.Switch
    handler: python
    options:
      show_root_heading: true
      show_source: true
      show_base: true

### Base classes

::: thunder.policy
    handler: python
    options:
      members:
        - Policy
        - MappingPolicy
      show_root_heading: true
      show_source: true
      show_base: true
