import typer
from piknlp.sentiment.cli import app as sentiment_app
from piknlp.category.cli import app as category_app
from piknlp.DB.cli import app as db_app

app = typer.Typer()
app.add_typer(sentiment_app, name="sentiment")
app.add_typer(category_app, name="category")
app.add_typer(db_app, name="db")

if __name__ == "__main__":
    app()