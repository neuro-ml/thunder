# MetricMonitor
This callback takes on computation and aggregation of the specified metrics.  

## Usage

### Loss
Despite the word `Metric` in the name, this callback also takes on logging of train
loss(es). It casts them by the following rules:  
- If `loss` is of type `torch.Tensor` - `{"loss": loss}` is logged.  
- If `loss` is a list or tuple, then it is logged as `{"i": loss_i}`.  
- If `loss` is a dict, then it is logged as is.  

:warning:  
> All Tensors are cast to numpy arrays.

At the end of epoch they are averaged and sent to logger.

### Metric Computation
Metrics are assumed to be received as tuple `(X, Y)`, where
**X** - batch of predictions, **Y** - batch of targets. 
Further process of computation depends on whether `Group` or `Single`
metrics are used. Also, there is no difference for MetricMonitor between 
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
from thunder.callbacks import MetricMonitor
from sklearn.metrics import accuracy_score

trainer = Trainer(callbacks=[MetricMonitor(group_metrics={"accuracy": accuracy_score})])
```

If you use any loggers (e.g. `Tensorboard` or `WandB`), `accuracy` will appear in them as follows:  
`val/accuracy` - validation metrics.  
`test/accuracy` - test metrics.

You can also use preprocessing functions as keys of the dictionary. It is 
covered in **Preprocessing** part in **Single Metrics** paragraph. Here is simple example
```python
from sklearn.metrics import accuracy_score, recall_score
# y is binary label
# x is e.g. a binary tensor and we want to know if there is any true value in it.
threshold = lambda y, x: (y, x.any())

group_metrics = {threshold: [accuracy_score, recall_score]}
```
Despite group metrics being calculated on collections of entries, __*preprocessing is applied individually*__.

### Single metrics
Single metrics are computed on each object separately and only then aggregated.
It is a common use case for tasks like segmentation or object detection.
#### Simple use case

```python
from thunder.callbacks import MetricMonitor
from sklearn.metrics import accuracy_score

trainer = Trainer(callbacks=[MetricMonitor(single_metrics={"accuracy": accuracy_score})])
```
MetricMonitor will log mean values by default. But you can add custom aggregations as well.
#### Custom aggregations
Let see what can be done if we want to log `std` of metrics as well as mean values.

```python
import numpy as np
from thunder.callbacks import MetricMonitor
from sklearn.metrics import accuracy_score

aggregate_fn = np.std

metric_monitor = MetricMonitor(single_metrics={"accuracy": accuracy_score},
                              aggregate_fn=aggregate_fn)

trainer = Trainer(callbacks=[metric_monitor])
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
MetricMonitor can accept `str` or `List[str]` as `aggregate_fn`, 
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

threshold = lambda y, x: (y > 0.5, x)

single_metrics = {threshold: [accuracy_score, recall_score]} 
# or
single_metrics = {threshold: {"acc": accuracy_score, "rec": recall_score}}
# or
single_metrics = {threshold: recall_score}
...
```
In the example above, `accuracy_score` and `recall_score` are computed on each case separately 
(e.g. like in semantic segmentation task). 
Preprocessing functions are applied on each entry separately in both `single_megtrics` and `group_metrics`.
#### Individual Metrics
While computing `single_metrics`, one may appear in need of knowledge of metrics on each case.
For this particular problem, the callback provides its users with `log_individual_metrics`
flag. Being set to `True` it forces the callback to store table of metrics in the following format:

| Name         | metric1    | metric2     |  
|--------------|------------|-------------|
| batch_idx0_0 |     some_value      | some_value |  
| batch_idx0_1 | some_value          | some_value    |  
| ...          | ...        | ...         |
| batch_idxn_m | some_value | some_value  |

For each set (e.g. `val`, `test`) and each `dataloader_idx`, MetricMonitor stores separate table.  
By default aforementioned tables are saved to `default_root_dir` of lightning's Trainer, in the format of
`set_name/dataloader_idx.csv` (e.g. `val/dataloader_0.csv`).  
If loggers you use have method `log_table` (e.g. `WandbLogger`), 
then this method will receive key and each table in the format of `pd.DataFrame`.  
Code from `metric_monitor.py`:
```python
logger.log_table(f"{key}/dataloader_{dataloader_idx}", dataframe=dataframe)
```
where key is the current state of trainer (`val` or `test`).  

Since lightning allows to use `batch_idx`, these indexes are used for metrics dataframes.
But there can be more than one object in batch. To overcome this issue we iterate over batch
and mark each object with the next index: 
```python
for i, object in enumerate(batch):
    object_idx = f"{batch_idx}_{i}"
```
If all batches consist of single object, then `"_{i}"` is removed.


## Reference
::: thunder.callbacks.metric_monitor.MetricMonitor
    handler: python
    options:
      members:
        - init
      show_root_heading: true
      show_source: false
