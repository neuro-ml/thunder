from pathlib import Path

import pytest


@pytest.fixture
def temp_dir(tmpdir):
    return Path(tmpdir)
