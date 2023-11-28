# Overview

Lightning Callbacks allow you to modify your training pipelines.  
For extra information see [this](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html).

## Thunder Callbacks
| Name                                   | Description                                           |
|----------------------------------------|-------------------------------------------------------|
| [MetricMonitor](./metric_monitor)        | Computes metrics and logs them                        |
| [TimeProfiler](./time_profiler)        | Logs the time of each LightningModule's step          |
| [FailOnInterrupt](./fail_on_interrupt) | Forces lightning Trainer to fail on KeyboardInterrupt |
