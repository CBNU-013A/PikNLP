# piknlp/DB/cli.py

import typer
from piknlp.DB.loadReviews import export_reviews_to_csv

app = typer.Typer()

@app.command()
def load_reviews(
    min_length: int = typer.Option(10, help="Minimum length of reviews to load"),
    max_reviews: int = typer.Option(None, help="Maximum number of reviews to load"),
    output_path: str = typer.Option("data/sentiment/raw/reviews.csv", help="Output path for the reviews"),
):
    typer.echo("Loading reviews...")
    export_reviews_to_csv(min_length=min_length, max_reviews=max_reviews, output_path=output_path)