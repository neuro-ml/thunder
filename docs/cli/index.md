# Command Line Interface
#### Requirements: [lazycon](https://github.com/maxme1/lazycon)  
Thunder provides its users with CLI to bring convenience and comfort into experiment
building and execution routine.

## Examples 
```bash
# 1. Buiding an experiment and overwriting some parameters
thunder build /path/to/config /path/to/experiment -u batch_size=2 -u lr=0.001

# 2. Running previously build experiment
thunder run /path/to/experiment --backend slurm -c 4 -r 100G 

# 3. Preparing new backend
thunder backend add my_awesome_back backend=slurm ram=100 cpu=4 gpu=1 partition=partition_name
# 4. Running experiment with `my_awesome_backend` 
thunder run /path/to/experiment --backend my_awesome_back
# 5. Setting `my_awesome_back` as the default
thunder backend set my_awesome_back
# 6. Same as 4
thunder run /path/to/experiment
```

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
If `/path/to/experiment` already exists, thunder raises error. 
In order to overwrite existing directory use `--overwrite` / `-o` flags.

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
Create `~/.config/thunder/backends.yml` (you can run `thunder backend list` in your terminal, 
required path will be at the title of the table) in you home directory. 
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

### Add, Set, List, Remove 
`thunder` CLI provides its users with built-in tools for managing their [backends](./#backend).

| Command                  | Description                                                       |
|--------------------------|-------------------------------------------------------------------|
| `thunder backend add`    | Add run config to the list of available configs.                  |
| `thunder backend list`   | Show parameters of specified backend(s).                          |
| `thunder backend remove` | Delete backend from list.                                         |
| `thunder backend set`     | Set specified backend from list of available backends as default. |

#### Examples
##### add

```bash
thunder backend add run_config_name backend=slurm ram=100 cpu=4 gpu=1 partition=partition_name
```
If specified name already exists, you can use `--force` flag in order to overwrite it.  

##### set
```bash
thunder backend set SOMENAME
thunder backend list
```

##### list 
```bash
thunder backend list NAME1 NAME2
*shows backends with specified names*

thunder backend list
*shows all backends*
```

##### remove
```bash
thunder backend remove SOMENAME
```


## Placeholders
Some loggers and other tools in your experiment may require name 
of the experiment. We find it convenient to use name of the folder you 
build your experiment into as the name of the experiment for loggers.   
Example with `WandbLogger`:
```python
from lightning.pytorch.loggers import WandbLogger
from thunder.placeholders import ExpName, GroupName

logger = WandbLogger(name=ExpName, group=GroupName)
```
In this case `GroupName` - name of the folder with built experiment and
`ExpName` - name of the split.

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
In [sweep config](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#sweep-configuration-examples) instead of `train.py` one should specify `experiment.config`.