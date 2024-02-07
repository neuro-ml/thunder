> You saw the lightning. Now it's time to hear the thunder 🌩️

# Thunder

🌩️ The Deep Learning framework based on [Lightning](https://lightning.ai/)

## Install

```bash
pip install thunder
```

:warning:  
> Currently thunder is not published on pypi. Install it via git clone.  

## Start experimenting
Many frameworks provide you with interfaces for your models and training pipelines, but we 
have yet to see any tools for creating whole experiment.

With :thunder: it's as simple as 1, 2, 3:

1. Create a config (e.g. `base.config`):
    ```python
    from myproject import MyDataset, MyModule
    from lightning import Trainer
    from torch.utils.data import DataLoader
    
    # these 3 fields are required
    train_data = DataLoader(MyDataset())
    module = MyModule()
    trainer = Trainer()
    ```

2. Build the experiment:
   ```shell
   thunder build base.config /path/to/some/folder
   ```
3. Run it
    ```shell
    thunder run /path/to/some/folder
    ```

Also, 2 and 3 can be combined into a single command:
```shell
thunder build-run base.config /path/to/some/folder
```

<div class="termy">

```console
$ thunder build base.config /path/to/some/folder
```

</div>


## Core Features
- ### [ThunderModule](./core/thunder_module)
- ### [MetricMonitor](./callbacks/metric_monitor)
- ### [Experiment configs](./configs)
- ### [CLI & Integrations with WandB](./cli)
