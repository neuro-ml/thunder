# Policy
Policies are objects that define how some value changes through time.
Good example of them is Learning Rate Schedulers.  

## Learning Rate Schedulers
Contrary to default PyTorch Learning Rate schedulers, ours does not require an
optimizer to be passed during initialization.

## Thunder Schedulers

| Name                                                 | Description                                             |
|------------------------------------------------------|---------------------------------------------------------|
| [Multiply](./lr_schedulers/#thunder.policy.Multiply) | Multiplies lr on each step by specified factor.         |
| [Schedule](./lr_schedulers/#thunder.policy.Schedule) | Assigns lr values according to specified callable.      |
| [Switch](./lr_schedulers/#thunder.policy.Switch)     | Assigns lr values according to specified dict schedule. |

See [Learning Rate Schedulers docs](./lr_schedulers.md) 





