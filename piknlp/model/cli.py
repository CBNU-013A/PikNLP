# piknlp/model/cli.py

import typer
from rich import print
from pathlib import Path
from dotenv import load_dotenv
import os
from piknlp.model.train import SentimentTrainer
from piknlp.model.test import SentimentTester, CategoryTester
from piknlp.common.config import Config
from piknlp.model.multi_train import MultiHeadTrainer

app = typer.Typer()

# Load .env file
load_dotenv()

@app.callback()
def main(
    ctx: typer.Context,
    config_path: Path = typer.Option(
        Path("config/train_config.yaml"), "--config", "-c", help="YAML 설정 파일 경로"
    ),
):
    ctx.obj = ctx.obj or {}
    # 도움말 호출 등에서도 콜백은 실행되므로, 여기서는 경로만 저장하고
    # 실제 Config 로드는 각 커맨드 내부에서 수행합니다.
    ctx.obj["config_path"] = config_path

@app.command("multi_train")
def multi_train(ctx: typer.Context):
    """
    Train the model with multiple heads.
    """
    config = Config(ctx.obj["config_path"]) 
    trainer = MultiHeadTrainer(config)
    trainer.train()

@app.command("train")
def train(ctx: typer.Context):
    """
    Train the model.
    """
    config = Config(ctx.obj["config_path"]) 
    if config.task == "sentiment":
        trainer = SentimentTrainer(config)
    elif config.task == "category":
        trainer = MultiHeadTrainer(config)
    else:
        raise ValueError(f"Invalid task: {config.task}")
    trainer.train()

@app.command("test")
def test(ctx: typer.Context):
    """
    Test the model.
    """
    config = Config(ctx.obj["config_path"]) 
    if config.task == "sentiment":
        tester = SentimentTester(config)
    elif config.task == "category":
        tester = CategoryTester(config)
    else:
        raise ValueError(f"Invalid task: {config.task}")
    tester.test()

@app.command("upload")
def upload_model(
    ctx: typer.Context,
    model_path: Path = typer.Argument(..., help="Path to the model directory to upload"),
):
    """
    Upload the model to Huggingface Hub.
    """
    # Huggingface 토큰 확인
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("[red bold]Error:[/red bold] HF_TOKEN not found in .env file")
        raise typer.Exit(code=1)
    
    # Check if the repository name is set
    config = Config(ctx.obj["config_path"]) 
    if not hasattr(config, "hf_repo_name") or not config.hf_repo_name:
        print("[red bold]Error:[/red bold] hf_repo_name not found in config")
        raise typer.Exit(code=1)
    
    print(f"[green]Uploading model from {model_path} to {config.hf_repo_name}...[/green]")
    
    # Check if the model path exists
    if not model_path.exists():
        print(f"[red bold]Error:[/red bold] Model path {model_path} does not exist.")
        raise typer.Exit(code=1)
    
    # Create a trainer instance (only for upload functionality)
    trainer = SentimentTrainer(config)
    
    try:
        trainer.upload_to_huggingface(model_path, config.hf_repo_name, hf_token)
        print("[green]✅ Model successfully uploaded to Huggingface Hub![/green]")
    except Exception as e:
        print(f"[red bold]Error:[/red bold] Failed to upload model: {str(e)}")
        raise typer.Exit(code=1)