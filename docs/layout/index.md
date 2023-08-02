# Layout

Layout instances are responsible for splitting your datasets and managing
which data fold is used for each experiment.
They also check reproducibility of your data splits.

## Thunder Layouts

| Name                                                      | Description                                        |
|-----------------------------------------------------------|----------------------------------------------------|
| [Split](./splits/#thunder.layout.split.Split)             | Layout for K fold cross-validation                 |
| [SingleSplit](./splits/#thunder.layout.split.SingleSplit) | Layout with several sets (e.g. train + val + test) |


All Layout subclasses follow common interface
::: thunder.layout.interface.Layout
    handler: python
    options:
      show_root_heading: true
      show_source: true
      show_base: true
