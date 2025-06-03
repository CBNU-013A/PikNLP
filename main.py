import typer
from piknlp.generator.cli import app as generator_app
from piknlp.model.cli import app as model_app
from piknlp.DB.cli import app as db_app

app = typer.Typer()
app.add_typer(generator_app, name="dataset")
app.add_typer(model_app, name="model")
app.add_typer(db_app, name="db")

if __name__ == "__main__":
    app()