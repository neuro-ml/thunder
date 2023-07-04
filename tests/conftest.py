from pathlib import Path

import pytest


@pytest.fixture
def temp_dir(tmpdir):
    return Path(tmpdir)


@pytest.fixture
def dumb_config():
    return Path(__file__).parent.resolve() / "assets" / "dumb.config"
