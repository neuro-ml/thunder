# Command Line Interface
#### Requirements: [lazycon](https://github.com/maxme1/lazycon)  
Thunder provides its users with CLI to bring convenience and comfort into experiment
building and execution routine.

For any help you can use
```bash
thunder --help
```
## Building an experiment
In order to build an experiment, you can execute the follwing command:
```bash
thunder build /path/to/config /path/to/experiment
```
It will create a folder with built configs in it.  
### Overriding config entries
While conducting experiments one can find themselves in constant need of
changing significant number of parameters. But it is not convenient to always do
it via IDE or any other code editor. 
Thunder gives an ability to override the values while building an experiment.

If in your config you have
```python
batch_size = 1
lr = 0.01
```
You can override it using `-u` flag:
```bash
thunder build /path/to/config /path/to/experiment -u batch_size=2 -u lr=0.001 
```
`batch_size` and `lr` will be assigned 2 and 0.001 respectively.

## Running an experiment
You can run built experiment by executing the next command:
```bash
thunder run /path/to/experiment
```
Under the hood thunder extracts necessary entries (e.g. model and trainer)
from your built config and executes `trainer.run(model, train_data, ...)`.

## Backend
As default options Thunder supports several backends:
- cli (default)
- slurm

You can switch between them by specifying `--backend` flag. 
```bash
thunder run /path/to/experiment/ --backend slurm -c 4 -r 100G 
```
The command shown above will run SLURM job with 4 CPUs and 100G of RAM.

### Predefined run configs
You can predefine run configs to avoid reentering the same flags.
Create `.config/thunder/backends.yml` in you home directory. 
Now you can specify config name and its parameters:
```yaml
run_config_name:
  backend: slurm
  config:
    ram: 100G
    cpu: 4
    gpu: 1
    partition: partition_name
```
In order to run an experiment with predefined parameters, 
use `--backend` flag as in previous section:
```bash
thunder run /path/to/experiment/ --backend run_config_name
```
You can overwrite parameters if you want to (e.g. 8 CPUs instead of 4):
```bash
thunder run /path/to/experiment/ --backend run_config_name -c 8
```


## WandB Sweeps integration
[WandB](https://www.wandb.com) has hyperparameters tuning system called [Sweeps](https://docs.wandb.ai/guides/sweeps).
Sweeps allow you to run multiple experiment with predefined grid of parameters and
compare run results. However, we find default sweep execution system very inconvenient
when it comes to running experiments on cluster.

After running a few experiments with 
[WandB Logger](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb), 
you can create sweep configuration. 
WandB will give a command `wandb agent project/sweep_id`.
You can copy it and paste it into the following command:  
```bash
thunder PASTE_HERE /path/to/config /path/to/experiment 
```