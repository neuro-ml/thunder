import pytest
from lazycon import Config
from sklearn.model_selection import KFold

from thunder.layout import FixedSingleSplit, FixedSplit, Node, Single, SingleSplit, Split


@pytest.mark.parametrize('layout', (
    Single(), SingleSplit([1, 2, 3], train=1, test=2), Split(KFold(3), [1, 2, 3]),
))
def test_basic_properties(layout, temp_dir):
    text = 'a = 1\nb = 2\n'
    config = Config.loads(text)
    nodes = list(layout.build(temp_dir, config))
    assert set(map(type, nodes)) <= {Node}
    assert len({x.name for x in nodes}) == len(nodes)
    # make sure the base config hasn't changed
    assert config.dumps() == text
    assert (temp_dir / 'experiment.config').exists()


def test_single(temp_dir):
    layout = Single()
    config = Config()
    layout.build(temp_dir, config)
    # make sure the base config hasn't changed
    assert layout.load(temp_dir, None)[-1] == {}


def test_split(temp_dir):
    layout = Split(KFold(3, shuffle=True, random_state=0), [1, 2, 3])
    nodes = list(layout.build(temp_dir, Config()))
    assert len(nodes) == 3
    _, p, kw = layout.load(temp_dir, Node(name='0'))
    assert p == temp_dir / 'fold_0'
    assert layout.splits[0] == kw['split']
    assert set(kw) == {'fold', 'split'}
    layout.set(**kw)

    layout = Split(KFold(3, shuffle=True, random_state=0), [1, 2, 3], names=['train', 'test'])
    layout.set(**layout.load(temp_dir, Node(name='0'))[-1])
    assert (layout.train, layout.test) == (layout[0], layout[1]) == layout.splits[0]

    with pytest.raises(ValueError):
        layout = Split(KFold(3, shuffle=True, random_state=1), [1, 2, 3])
        layout.set(**layout.load(temp_dir, Node(name='0'))[-1])


def test_single_split(temp_dir):
    layout = SingleSplit([1, 2, 3], train=1, test=2)
    nodes = list(layout.build(temp_dir, Config()))
    assert len(nodes) == 0
    _, p, kw = layout.load(temp_dir, None)
    assert p == temp_dir
    assert set(kw) == {'split'}
    assert layout.split == kw['split']
    layout.set(**kw)

    layout = SingleSplit([1, 2, 3], train=1, test=2)
    layout.set(**layout.load(temp_dir, None)[-1])
    assert {'train': layout.train, 'test': layout.test} == layout.split

    with pytest.raises(ValueError):
        # check consistency error
        layout = SingleSplit([1, 2, 3], train=1, test=2, random_state=1)
        layout.set(**layout.load(temp_dir, None)[-1])

    # check if split consists of 255 cases and not 254 or smth else.
    layout = SingleSplit(list(range(255)), train=0.7, test=0.3)
    assert len(layout.train) + len(layout.test) == 255

    # check negative
    with pytest.raises(ValueError, match="non-negative"):
        SingleSplit([1, 2, 3], train=1, test=-1)


def test_fixed_split(temp_dir):
    layout = FixedSplit([[[1, 2, 3], [4, 5], [6, 7]]], "train", "val", "test")
    nodes = list(layout.build(temp_dir, Config()))
    assert len(nodes) == 1
    _, p, kw = layout.load(temp_dir, Node(name='0'))
    assert p == temp_dir / 'fold_0'
    assert layout.splits[0] == kw['split']
    assert set(kw) == {'fold', 'split'}
    layout.set(**kw)

    _check_attributes(layout.names, layout)


@pytest.mark.parametrize("layout", [FixedSingleSplit([[1], [2], [3]], "train", "val", "test"),
                                    FixedSingleSplit({"train": [1], "val": [2], "test": [3]})])
def test_fixed_single_split(layout, temp_dir):
    nodes = list(layout.build(temp_dir, Config()))
    assert len(nodes) == 0
    _, p, kw = layout.load(temp_dir, None)
    assert p == temp_dir
    assert set(kw) == {'split'}
    assert layout.split == kw['split']
    layout.set(**kw)

    _check_attributes(layout.split.keys(), layout)


def _check_attributes(keys, layout):
    for k in keys:
        assert getattr(layout, k)
