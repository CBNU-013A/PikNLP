# piknlp/sentiment/cli.py

import typer
from rich import print
from pathlib import Path
from dotenv import load_dotenv
import os
from .generator import Generator
from .train import SentimentTrainer
from .test import SentimentTester
from piknlp.common.config import Config

app = typer.Typer()

# .env 파일 로드
load_dotenv()

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

@app.command("train")
def train():
    trainer = SentimentTrainer(config)
    trainer.train()

@app.command("test")
def test():
    tester = SentimentTester(config)
    tester.test()

@app.command("upload")
def upload_model(
    model_path: Path = typer.Argument(..., help="Path to the model directory to upload"),
):
    """
    학습된 모델을 Huggingface Hub에 업로드합니다.
    """
    # Huggingface 토큰 확인
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("[red bold]Error:[/red bold] HF_TOKEN not found in .env file")
        raise typer.Exit(code=1)
    
    # 레포 이름 확인
    if not hasattr(config, "hf_repo_name") or not config.hf_repo_name:
        print("[red bold]Error:[/red bold] hf_repo_name not found in config")
        raise typer.Exit(code=1)
    
    print(f"[green]Uploading model from {model_path} to {config.hf_repo_name}...[/green]")
    
    # 모델 경로 확인
    if not model_path.exists():
        print(f"[red bold]Error:[/red bold] Model path {model_path} does not exist.")
        raise typer.Exit(code=1)
    
    # 트레이너 인스턴스 생성 (업로드 기능만 사용)
    trainer = SentimentTrainer(config)
    
    try:
        trainer.upload_to_huggingface(model_path, config.hf_repo_name, hf_token)
        print("[green]✅ Model successfully uploaded to Huggingface Hub![/green]")
    except Exception as e:
        print(f"[red bold]Error:[/red bold] Failed to upload model: {str(e)}")
        raise typer.Exit(code=1)