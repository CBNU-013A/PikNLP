# piknlp/generator/cli.py

import typer
from rich import print
from dotenv import load_dotenv
from piknlp.generator.generator import Generator
from piknlp.common.config import Config

app = typer.Typer()

# .env 파일 로드
load_dotenv()

@app.callback()
def main(
    ctx: typer.Context,
    config_path: str = typer.Option(
        "config/train_config.yaml", "--config", "-c", help="YAML 설정 파일 경로"
    ),
):
    ctx.obj = ctx.obj or {}
    ctx.obj["config_path"] = config_path

@app.command("generate")
def generate(ctx: typer.Context):
    """
    Generate the dataset.
    """
    print("[green]Running dataset generation...[/green]")
    config = Config(ctx.obj["config_path"])
    generator = Generator(config)
    generator.generate_dataset()

@app.command("split")
def split(
    ctx: typer.Context,
    include_dev: bool = typer.Option(False, help="Include dev set when splitting"),
):
    """
    Split the dataset into train/test/dev sets.
    """
    config = Config(ctx.obj["config_path"])
    generator = Generator(config)
    generator.split_dataset(include_dev=include_dev)