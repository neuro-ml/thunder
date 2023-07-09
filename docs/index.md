> You saw the lightning. Now it's time to hear the thunder
# Thunder
The Deep Learning framework based on Lightning.

## Installation
You can install from pypi  
```bash
pip install thunder
```
or directly from GitHub  
```bash
git clone https://github.com/neuro-ml/thunder.git
cd thunder && pip install -e .
```

## Start experimenting

It's as simple as 1, 2, 3:

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

## Core Features
- ### [ThunderModule](./core/thunder_module)
- ### [MetricLogger](./callbacks/metric_logger)
- ### [Experiment configs](./configs)
- ### [CLI & Integrations with WandB](./cli)
