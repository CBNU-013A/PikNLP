# piknlp/DB/cli.py

import typer
from piknlp.DB.loadReviews import export_reviews_to_csv
from piknlp.common.config import Config

app = typer.Typer()

@app.callback()
def main(
    ctx: typer.Context,
    config_path: str = typer.Option(
        "config/train_config.yaml", "--config", "-c", help="YAML 설정 파일 경로"
    ),
):
    ctx.obj = ctx.obj or {}
    ctx.obj["config_path"] = config_path

@app.command()
def load_reviews(
    ctx: typer.Context,
    min_length: int = typer.Option(10, help="Minimum length of reviews to load"),
    max_reviews: int = typer.Option(None, help="Maximum number of reviews to load"),
    output_path: str = typer.Option(None, help="Output path for the reviews"),
):
    typer.echo("Loading reviews...")
    config = Config(ctx.obj["config_path"])
    resolved_output = output_path or f"data/{config.task}/raw/reviews.csv"
    export_reviews_to_csv(min_length=min_length, max_reviews=max_reviews, output_path=resolved_output)