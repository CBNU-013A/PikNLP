# scripts/model.py
# 모델 학습 / 평가

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.finetune_runner import ModelRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model on labeled ACSC data")
    parser.add_argument("--config_file", type=str, required=True, help="Config File")
    parser.add_argument("--config_dir", type=str, default="model/config", help="Config Directory")
    parser.add_argument("--mode", type=str, required=True, help="train, eval, test")

    cli_args = parser.parse_args()
    ModelRunner.from_cli_args(cli_args)