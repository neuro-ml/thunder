import typer
from typing import Optional, Annotated
from pydantic import BaseModel

class TestConfig(BaseModel):
    ram: Annotated[Optional[str], typer.Option(
        None, '--ram', '--mem', '-r',
        help='RAM test option'
    )] = None
    

app = typer.Typer()

@app.command(cls=TestConfig)
def test(config: TestConfig):
    typer.echo(f"RAM: {config.ram}")

if __name__ == "__main__":
    app()