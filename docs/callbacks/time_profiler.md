## TimeProfiler
Lightning Callback which allows you to measure the time each step takes and log it during the training process.


## Logged values
TimeProfiler logs the following steps:

| Name                 | Logged by default  | Description                                                            |  
|----------------------|--------------------|------------------------------------------------------------------------|
| train batch          | :white_check_mark: | Time taken by forward, optimizer step, and backward during train step. |  
| validation batch     | :white_check_mark: | Time taken by forward during validation step.                          |  
| train epoch          | :white_check_mark: | Time taken by train epoch without validation.                          |
| validation epoch     | :white_check_mark: | Time taken by validation epoch.                                        |
| avg train downtime\* | :white_check_mark: | Average downtime in training step.                                     |
| avg val downtime     | :white_check_mark: | Average downtime in validation step.                                   |
| backward             | :x:                | Time taken by backprop.                                                |
| optimizer step       | :x:                | Time taken by optimizer.                                               |
| total train downtime | :x:                | Total downtime in training epoch.                                      |
| total val downtime   | :x:                | Total downtime in validation epoch.                                    |
 

*Downtime - the process during which model does not work (e.g. data loader is working now)  

## Usage
```python
from thunder.callbacks import TimeProfiler
from lightning import Trainer

# logs default keys and in addition backward and optimizer step
trainer = Trainer(callbacks=[TimeProfiler("backward", "optimizer step")])
```


## Reference
::: thunder.callbacks.time_profiler.TimeProfiler
    handler: python
    options:
      members:
        - init
      show_root_heading: true
      show_source: false
