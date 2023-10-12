from typing import Dict, Optional, Sequence, Union

from more_itertools import collapse, zip_equal

from .split import SingleSplit, Split


class FixedSplit(Split):
    def __init__(self, splits: Sequence,
                 names: Optional[Sequence[str]] = None):
        """
        Creates experiment layout from given split.
        Parameters
        ----------
        splits: Sequence
            Split of data.
        names: Optional[Sequence[str]]
            Names of folds, e.g. 'train', 'val', test'.
        """
        if names is not None:
            if len(set(names)) != len(names):
                raise ValueError(f"Names of splits are not unique: {names}")
            if len(splits[0]) != len(names):
                raise ValueError(f"Got {len(splits[0])} and {len(names)} fold names: {names}")

        self.entries = sorted(set(collapse(splits)))
        self.splits = [tuple(fold) for fold in splits]
        self.names = names
        self.fold: Optional[int] = None


class FixedSingleSplit(SingleSplit):
    def __init__(self, split: Union[Sequence, Dict[str, Sequence]],
                 names: Optional[Sequence[str]] = None):
        """
        Creates single fold experiment from given split.
        Parameters
        ----------
        split: Union[Sequence, Dict[str, Sequence]]
            split of data
        Names of folds, e.g. 'train', 'val', test'. If data is of type `dict`,
        then it is not required.
        """
        if isinstance(split, dict):
            names = split.keys()  # from python3.7 order is guaranteed.
            split = split.values()

        if len(names) != len(split):
            raise ValueError("Difference in number of splits and number of names: "
                             f"{len(split)} and {len(names)}")

        self.split = dict(zip_equal(names, split))
