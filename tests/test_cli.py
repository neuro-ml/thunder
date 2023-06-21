import re

from lazycon import Config
from typer.testing import CliRunner

from thunder.cli.entrypoint import app, populate


populate()
runner = CliRunner()


def test_build(temp_dir):
    experiment = temp_dir / 'exp'
    config = temp_dir / 'x.config'
    # lang=Python
    Config.loads('''
from thunder.layout import Single
layout = Single()
    ''').dump(config)

    result = runner.invoke(app, ['build', str(config), str(experiment)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ['build', str(config), str(experiment)])
    assert result.exit_code == 1
    assert re.match('Cannot create an experiment in the folder ".*", it already exists\n', result.output)
