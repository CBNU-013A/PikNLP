# piknlp/sentiment/cli.py

import typer
from rich import print
from .generator import Generator
from piknlp.common.config import Config

app = typer.Typer()

config = Config()

@app.command("generate")
def generate(
    split: bool = typer.Option(False, help="Split dataset into train/test"),
    dev: bool = typer.Option(False, help="Include dev set when splitting"),
):
    # 유효성 검사
    if dev and not split:
        print("[red bold]Error:[/red bold] --dev can only be used with --split.")
        raise typer.Exit(code=1)

    print("[green]Running dataset generation...[/green]")
    generator = Generator(config)
    generator.generate_dataset()
    if split:
        print(f"[yellow]Splitting dataset... dev={'included' if dev else 'excluded'}[/yellow]")
        generator.split_dataset(include_dev=dev)