# Layout

Layout instances are responsible for splitting your datasets and managing
which data fold is used for each experiment.
They also check reproducibility of your data splits.

## Usage

Layout is a cornerstone of every built experiment. Via layout one can define structure of 
their data splits.  

After building an experiment one may want to extract data of created splits. 
Heretofore it was done by manually loading the ids. Thunder's Layouts provides interface
for extracting them by simply loading experiment config. 
```python
import lazycon
cfg = lazycon.load("/path/to/experiment.config")
cfg.layout.set(fold=0)
train_ids = cfg.layout.train # cfg.layout.SPLIT_NAME 
```

## Thunder Layouts

| Name                                                               | Description                                            |
|--------------------------------------------------------------------|--------------------------------------------------------|
| [Split](./splits/#thunder.layout.split.Split)                      | Layout for K fold cross-validation                     |
| [SingleSplit](./splits/#thunder.layout.split.SingleSplit)          | Layout with several sets (e.g. train + val + test)     |
| [FixedSplit](./fixed/#thunder.layout.fixed.FixedSplit)             | Creates layout from predefined K-fold   split          |
| [FixedSingleSplit](./fixed/#thunder.layout.fixed.FixedSingleSplit) | Creates single fold layout from predefined data split. |

All Layout subclasses follow common interface
::: thunder.layout.interface.Layout
    handler: python
    options:
      show_root_heading: true
      show_source: true
      show_base: true
