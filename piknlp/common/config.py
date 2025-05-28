# piknlp/common/config.py

import yaml
from pathlib import Path
from piknlp.common.logger import get_logger

class Config:
    def __init__(self, config_path: Path = Path("config/config.yaml")):
        self.logger = get_logger(__name__)

        self.logger.info(f"Loading config from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.logger.info(f"Config loaded: {cfg}")
        
        self.llm: str = cfg.get("llm", "exaone3.5")
        self.data_dir: Path = Path(cfg.get("data_dir", "data"))
        self.raw_data_dir: Path = self.data_dir / "raw"
        self.interim_data_dir: Path = self.data_dir / "interim"
        self.processed_data_dir: Path = self.data_dir / "processed"
        self.category: list[str] = cfg.get("category", [])
        self.train_ratio: float = cfg.get("train_ratio", 0.8)
        self.dev_ratio: float = cfg.get("dev_ratio", 0.1)
        self.seed: int = cfg.get("seed", 42)
        self.num_workers: int = cfg.get("num_workers", 5)

    def __getitem__(self, key: str):
        return getattr(self, key)
    
    def __repr__(self):
        return f"Config(llm={self.llm}, data_dir={self.data_dir}, raw_data_dir={self.raw_data_dir}, interim_data_dir={self.interim_data_dir}, processed_data_dir={self.processed_data_dir}, category={self.category}, train_ratio={self.train_ratio}, dev_ratio={self.dev_ratio}, seed={self.seed})"
