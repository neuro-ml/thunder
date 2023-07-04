# Lazycon
Thunder embraces the power of [lazycon](https://github.com/maxme1/lazycon)
allowing you to build configs for your 
experiments. 
## Config structure
Correct config should contain the following objects:

| Name                 | Required           | Description                          |  
|----------------------|--------------------|--------------------------------------|
| `trainer`            | :white_check_mark: | Lightning Trainer instance.          |  
| `module`             | :white_check_mark: | LightningModule instance.            |  
| `train_data`         | :white_check_mark: | Loader of training data.             |
| `val_data`           | :x:                | Loader of validation data.           |
| `test_data`          | :x:                | Loader of test data.                 |
| `datamodule`           | :x:                | LightningDataModule instance, replaces `train_data`, `val_data` and `test_data` if specified. |
 

## Executing a config
Thunder has its own **Command Line Interface**, 
about which you can read [here](../cli).
