import numpy as np

from thunder.callbacks import SeedEverything


def test_random():
    SeedEverything(42)
    x = np.random.randn(3)
    SeedEverything(42)
    assert np.all(x == np.random.randn(3))
    SeedEverything(0xBadCafe)
    assert not np.all(x == np.random.randn(3))


def test_state_dict():
    callback = SeedEverything(42, False)
    state_dict = callback.state_dict()
    new_callback = SeedEverything(34, True)
    new_callback.load_state_dict(state_dict)
    assert callback.__dict__ == new_callback.__dict__
