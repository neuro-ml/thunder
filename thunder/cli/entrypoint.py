from . import main as _main  # noqa
from .backend_cli import backend_app
from .main import app
from .wandb import wand_app


app.add_typer(wand_app)
app.add_typer(backend_app)


def main():
    app()
