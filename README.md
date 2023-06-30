[![docs](https://img.shields.io/badge/-docs-success)](https://neuro-ml.github.io/thunder/)
![License](https://img.shields.io/github/license/neuro-ml/thunder)
[![codecov](https://codecov.io/gh/neuro-ml/thunder/branch/master/graph/badge.svg)](https://codecov.io/gh/neuro-ml/thunder)
[![pypi](https://img.shields.io/pypi/v/thunder?logo=pypi&label=PyPi)](https://pypi.org/project/thunder/)

> > You saw the lightning. Now it's time to hear the thunder 🌩️

# Thunder

🌩️ The Deep Learning framework based on [Lightning](https://lightning.ai/)

## Install

```bash
pip install thunder
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

## More advanced stuff

See [our docs](https://neuro-ml.github.io/thunder/) for a full list of neat things `thunder`🌩️ can do for you