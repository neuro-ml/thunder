from pathlib import Path

from lazycon import Config
from pytorch_lightning import LightningModule, Trainer
from typer import Typer

app = Typer()


@app.command()
def start(config: Path):
    config = Config.load(config)
    # TODO: match by type rather than name?
    module: LightningModule = config.module
    trainer: Trainer = config.trainer
    # TODO: move get(..., default) to lazycon
    trainer.fit(module, config.train_data, getattr(config, 'val_data', None))


@app.command()
def build(config: Path, experiment: Path):
    # TODO: override
    root = Path(experiment)
    # TODO: permissions
    root.mkdir(parents=True)
    built_config = experiment / 'experiment.config'
    Config.load(config).dump(built_config)
    return built_config


@app.command()
def run(config: Path, experiment: Path):
    new_config = build(config, experiment)
    start(new_config)


def main():
    app()
