# Lazycon
Thunder embraces the power of [lazycon](https://github.com/maxme1/lazycon)
allowing you to build configs for your 
experiments. 
## Config structure
Correct config should contain the following objects:

| Name           | Required           | Description                                                                                                     |  
|----------------|--------------------|-----------------------------------------------------------------------------------------------------------------|
| `trainer`      | :white_check_mark: | Lightning Trainer instance.                                                                                     |  
| `module`       | :white_check_mark: | LightningModule instance.                                                                                       |  
| `train_data`   | :white_check_mark: | Loader of training data.                                                                                        |
| `val_data`     | :x:                | Loader of validation data.                                                                                      |
| `test_data`    | :x:                | Loader of test data.                                                                                            |
| `predict_data` | :x:                | Loader of test data.                                                                                            |
| `datamodule`   | :x:                | LightningDataModule instance, replaces `train_data`, `val_data` and `test_data` if specified.                   |
| `CALLBACKS`    | :x:                | Arbitrary function (list of functions e.g. `CALLBACKS=[seed_everything()]`) to be executed before run. |

After executing `thunder run` (see [Executing a config](./#executing-a-config)), thunder will extract 
necessary fields. If some optional field (e.g. `val_data`) is not provided, features 
dependent on it will not be used (e.g. no validation if `val_data` is not provided).

## Examples
Examples of configs can be seen [here](../examples)

## Executing a config
Thunder has its own **Command Line Interface**, 
about which you can read [here](../cli).

## Logging
All primitive values (e.g. int, float, tuples) are logged automatically via config parsing.
