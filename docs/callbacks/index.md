# Callbacks

Lightning Callbacks allow you to modify your training pipelines.  
For extra information see [this](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html).

## Thunder Callbacks
| Name                               | Description                                  |
|------------------------------------|----------------------------------------------|
| [MetricLogger](./metric_logger) | Computes metrics and logs them               |
| [TimeProfiler](./time_profiler)    | Logs the time of each LightningModule's step |
