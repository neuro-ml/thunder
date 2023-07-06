from . import main as _main  # noqa
from .main import app
from .wandb import wand_app


app.add_typer(wand_app)


def main():
    app()
