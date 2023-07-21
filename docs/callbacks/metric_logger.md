# MetricLogger
This callback takes on computation and aggregation of the specified metrics.  

## Usage

### Metric Computation
Metrics are assumed to be received as tuple `(X, Y)`, where
**X** - batch of predictions, **Y** - batch of targets. 
Further process of computation depends on whether `Group` or `Single`
metrics are used. Also, there is no difference for MetricLogger between 
`(X, Y)` and `((X,), (Y,))`.  
If your model has multiple outputs or requires multiple targets
(e.g. neural network with 2 heads.), the output is expected to be
`((X1, X2), (Y1, Y2))` (the most common way to represent such data in PyTorch), where **X1** is batch of model's first output.
In this case outputs will be recombined, so the first object
of the output will be `((x1, x2), (y1, y2))`, where **x1** is the first
element of batch `X1`.  

:warning:  
> Inside the callback outputs are swapped, so if LightningModule returns
(X, Y) then metrics will receive (Y, X).

### Group metrics
Group metrics are computed on the entire dataset.
For example, you want to compute classification accuracy on MNIST.
```python
from thunder.callbacks import MetricLogger
from sklearn.metrics import accuracy_score

trainer = Trainer(callbacks=[MetricLogger(group_metrics={"accuracy": accuracy_score})])
```

If you use any loggers (e.g. `Tensorboard` or `WandB`), `accuracy` will appear in them as follows:  
`val/accuracy` - validation metrics.  
`test/accuracy` - test metrics.

### Single metrics
Single metrics are computed on each object separately and only then aggregated.
It is a common use case for tasks like segmentation or object detection.
#### Simple use case
```python
from thunder.callbacks import MetricLogger
from sklearn.metrics import accuracy_score

trainer = Trainer(callbacks=[MetricLogger(single_metrics={"accuracy": accuracy_score})])
```
MetricLogger will log mean values by default. But you can add custom aggregations as well.
#### Custom aggregations
Let see what can be done if we want to log `std` of metrics as well as mean values.
```python
import numpy as np
from thunder.callbacks import MetricLogger
from sklearn.metrics import accuracy_score

aggregate_fn = np.std

metric_logger = MetricLogger(single_metrics={"accuracy": accuracy_score},
                             aggregate_fn=aggregate_fn) 

trainer = Trainer(callbacks=[metric_logger])
```
The mean values appear in loggers with no additional keys. 
MetricCallback will try to infer the name of an aggregating function
and use it as an additional key.

`val/accuracy` - validation mean accuracy.  
`val/std/accuracy` - validation accuracy std.  
`test/accuracy` - test mean accuracy.  
`test/std/accuracy` - test accuracy std.

`aggregate_fn` can also be specified as follows:

```python
import numpy as np

aggregate_fn = [np.std, np.median]
aggregate_fn = [np.std, "median", "max", "min"]
aggregate_fn = {"zero": lambda x: x[0]}
```
MetricLogger can accept `str` or `List[str]` as `aggregate_fn`, 
in this format it supports the following metrics:

| Name     | Function    |  
|----------|-------------|
| "median" | `np.median` |  
| "min"    | `np.min`    |  
| "max"    | `np.max`    |
| "std"    | `np.std`    |

#### Preprocessing
Sometimes metrics require some preprocessing. In this case, keys of `single_metrics` dict
must be callable objects.
```python
from sklearn.metrics import accuracy_score, recall_score

threshold = lambda x, y: (x > 0.5, y)

single_metrics = {threshold: [accuracy_score, recall_score()]} 
# or
single_metrics = {threshold: {"acc": accuracy_score, "rec": recall_score}}
# or
single_metrics = {threshold: recall_score}
...
```

## Reference
::: thunder.callbacks.metric_logger.MetricLogger
    handler: python
    options:
      members:
        - init
      show_root_heading: true
      show_source: false
