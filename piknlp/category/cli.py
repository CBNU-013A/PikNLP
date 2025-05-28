import typer
from rich import print

app = typer.Typer()

@app.command("generate-dataset")
def generate_dataset():
    print("Generating dataset...")

@app.command("train")
def train():
    print("Training...")