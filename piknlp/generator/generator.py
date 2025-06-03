# piknlp/generator/generator.py

import logging
import re
import pandas as pd
import json
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

from piknlp.common.config import Config
from piknlp.common.logger import get_logger
from piknlp.schema.generator import ReviewSample, SentimentList, ReviewLabel
from piknlp.generator.llm import call_ollama as call_llm
from piknlp.generator.llm import generate_dataset_prompt

class Generator:
    logger: logging.Logger = get_logger(__name__)
    
    def __init__(self, config: Config) -> None:
        self.config = config

    def generate_dataset(self) -> None:
        
        # Load raw data
        raw_file_path: Path = self.config.task_dir / "raw" / self.config.raw_data_file # /data/sentiment/raw/reviews.csv
        self.logger.info(f"🚀 Start generating dataset")
        self.logger.debug(f"Processing {raw_file_path}")
        reviews: list[str] = self._load_csv(raw_file_path)
        labeled_data: list[ReviewSample] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[green]리뷰 처리 중...", total=len(reviews))
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = [executor.submit(self._process_review, review) for review in reviews]
                for f in as_completed(futures):
                    sample = f.result()
                    labeled_data.append(sample)
                    progress.update(task, advance=1)
        
        save_path = self.config.task_dir / "dataset" / f"{self.config.task}.jsonl" # /data/sentiment/dataset/sentiment.jsonl
        self._save_json(labeled_data, save_path)
        self.logger.info("✅ Dataset generated successfully.")
        self.logger.debug(f"Saved {len(labeled_data)} samples to {save_path}")

    def split_dataset(self, include_dev: bool = False) -> None:
        dataset_path = self.config.task_dir / "dataset" / f"{self.config.task}.jsonl" # /data/sentiment/dataset/sentiment.jsonl
        self.logger.info(f"🔍 Splitting dataset from {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        self.logger.debug(f"Loaded {len(data)} samples from {dataset_path}")

        if include_dev:
            train_data, temp_data = train_test_split(
                data, test_size=(1 - self.config.split_ratio["train_ratio"]), random_state=self.config.split_ratio["seed"]
            )
            dev_size = self.config.split_ratio["dev_ratio"] / (1 - self.config.split_ratio["train_ratio"])
            dev_data, test_data = train_test_split(
                temp_data, test_size=(1 - dev_size), random_state=self.config.split_ratio["seed"]
            )
            splits = [("train", train_data), ("dev", dev_data), ("test", test_data)]
        else:
            train_data, test_data = train_test_split(
                data, test_size=(1 - self.config.split_ratio["train_ratio"]), random_state=self.config.split_ratio["seed"]
            )
            splits = [("train", train_data), ("test", test_data)]

        for name, split in splits:
            with open(self.config.task_dir / "dataset" / f"{name}.jsonl", "w", encoding="utf-8") as f: # /data/sentiment/dataset/{name}.jsonl
                for item in split:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
        self.logger.info("✅ Dataset split successfully.")
        self.logger.debug(f"Saved {len(train_data)} samples")

    def _load_csv(self, csv_path: Path, review_column: str = "Review") -> list[str]:
        df = pd.read_csv(csv_path)
        return df[review_column].dropna().tolist()

    def _save_json(self, data: list[dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(item.model_dump_json() + "\n")

    def _extract_json(self, text: str) -> str:
        try:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                json_str = match.group(0)
                # 키 변환
                data = json.loads(json_str)
                if "labels" in data:
                    data["sentiments"] = data.pop("labels")
                return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"⚠️ JSON extraction failed: {e}")
        return "{}"

    def _parse_labels(self, response: SentimentList) -> list[ReviewLabel]:
        try:
            assert isinstance(response, SentimentList), "Response type mismatch"
            return [
                ReviewLabel(category=cat, review=sentiment)
                for cat, sentiment in response.sentiments.items()
            ]
        except Exception as e:
            self.logger.error(f"Error parsing labels: {e}")
            return []

    def _process_review(self, review: str) -> ReviewSample:
        prompt = generate_dataset_prompt(review, self.config.category)
        raw_response = call_llm(prompt)
        clean_response = self._extract_json(raw_response)
        try:
            response_obj = SentimentList.model_validate_json(clean_response)
            labels = self._parse_labels(response_obj)
        except Exception as e:
            self.logger.error(f"Error parsing labels: {e}")
            labels = []
        return ReviewSample(sentence=review, label=labels)