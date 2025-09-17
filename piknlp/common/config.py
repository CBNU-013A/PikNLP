# piknlp/common/config.py

import yaml
from pathlib import Path
from piknlp.common.logger import get_logger

class Config:
    def __init__(self, config_path: Path = Path("config/train_config.yaml")):
        self.logger = get_logger(__name__)

        self.logger.info(f"🔍 Loading config from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.logger.debug(f"Config loaded: {cfg}")
        self.logger.info("✅ Config loaded successfully.")
        
        self.task: str = cfg.get("task", None)
        if self.task is None:
            raise ValueError("Task is not specified in the config file.")

        # Task Directory Setting
        self.task_dir: Path = Path("data") / self.task
        self.task_dir.mkdir(parents=True, exist_ok=True)

        # Dataset Generator parameters
        self.raw_data_file: str = cfg.get("raw_data_file", "reviews.csv")
        self.dataset_file: str = cfg.get("dataset_file", "dataset.jsonl")
        self.llm: str = cfg.get("llm_model_name", "exaone3.5")
        self.category: dict[str, list[str]] = cfg.get("category", {}) # Category for Sentiment Analysis labels
        self.label_list: list[str] = cfg.get("label_list", []) # Sentiment labels
        self.split_ratio: dict[str, float|int] = cfg.get("split", {})
        self.num_workers: int = cfg.get("num_workers", 5)

        # Train Settings
        self.cuda: bool = cfg.get("cuda", True)
        self.model_name: str = cfg.get("nlp_model_name", "electra")
        
        # Train files
        self.train_file: str = cfg.get("train_file", "train.jsonl")
        self.test_file: str = cfg.get("test_file", "test.jsonl")
        
        # Train parameters
        self.kfold_num: int = cfg.get("kfold_num", 5)
        self.kfold_seed: int = cfg.get("kfold_seed", 42)
        self.max_seq_len: int = cfg.get("max_seq_len", 128)
        self.train_batch_size: int = cfg.get("train_batch_size", 16)
        self.eval_batch_size: int = cfg.get("eval_batch_size", 32)
        self.max_steps: int = cfg.get("max_steps", 0)
        self.epochs: int = cfg.get("epochs", 3)
        self.gradient_accumulation_steps: int = cfg.get("gradient_accumulation_steps", 1)
        self.weight_decay: float = cfg.get("weight_decay", 0.01)
        self.learning_rate: float = float(cfg.get("learning_rate", 5e-5))
        self.adam_epsilon: float = float(cfg.get("adam_epsilon", 1e-8))
        self.warmup_proportion: float = cfg.get("warmup_proportion", 0.1)
        self.max_grad_norm: float = cfg.get("max_grad_norm", 1.0)
        self.early_stopping_patience: int = cfg.get("early_stopping_patience", 3)
        self.dropout_rate: float = cfg.get("dropout_rate", 0.1)
        
        # Evaluation parameters
        self.eval_num_workers: int = cfg.get("eval_num_workers", 4)
        
        # Huggingface Hub parameters
        self.hf_repo_name: str = cfg.get("hf_repo_name", None)

    def __getitem__(self, key: str):
        return getattr(self, key)
    
    def __repr__(self):
        return f"Config(task={self.task}, model_name={self.model_name}, category={self.category}, kfold_num={self.kfold_num})"
