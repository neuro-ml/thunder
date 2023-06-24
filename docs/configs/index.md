# Lazycon
Thunder embraces the power of [lazycon]() allowing you to build configs for your 
experiments. 
## Config structure
In order for the config to be correct it should contain the following objects:  
`trainer` - Lightning Trainer  
`model` - instance of LightningModule or [ThunderModule](../core/thunder_module.md)    
`train_data` - train dataloader  
`val_data` - val dataloader (Optional)  
`test_data` - test dataloader (Optional)

## Executing a config
Configs can be run just like usual python files:
```bash
python /path/to/config.config
```

But Thunder has its own command line interface, 
about which you can read [here](../cli).
