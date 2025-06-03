# piknlp/generator/cli.py

import typer
from rich import print
from dotenv import load_dotenv
from piknlp.generator.generator import Generator
from piknlp.common.config import Config

app = typer.Typer()

# .env 파일 로드
load_dotenv()

config = Config()

@app.command("generate")
def generate(
):
    """
    Generate the dataset.
    """
    print("[green]Running dataset generation...[/green]")
    generator = Generator(config)
    generator.generate_dataset()

@app.command("split")
def split(
    include_dev: bool = typer.Option(False, help="Include dev set when splitting"),
):
    """
    Split the dataset into train/test/dev sets.
    """
    generator = Generator(config)
    generator.split_dataset(include_dev=include_dev)