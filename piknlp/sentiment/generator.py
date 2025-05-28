# piknlp/sentiment/generator.py

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
from piknlp.common.schema import ReviewSample, SentimentList, ReviewLabel
from piknlp.sentiment.llm import call_ollama as call_llm
from piknlp.sentiment.llm import generate_dataset_prompt

class Generator:
    def __init__(self, config: Config) -> None:
        self.logger: logging.Logger = get_logger(__name__)
        self.config = config
        self.raw_data_dir: Path = self.config.raw_data_dir
        self.processed_data_dir: Path = self.config.processed_data_dir / "sentiment"

    def generate_dataset(self) -> None:
        file_path = self.raw_data_dir / "reviews.csv"
        self.logger.info(f"Start generating dataset")
        self.logger.info(f"Processing {file_path}")
        reviews: list[str] = self._load_csv(file_path)
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
        self._save_json(labeled_data, self.processed_data_dir / f"sentiment.jsonl")
        self.logger.info(f"Saved {len(labeled_data)} samples to {self.processed_data_dir}")

    def split_dataset(self, include_dev: bool = False) -> None:
        with open(self.processed_data_dir / "sentiment.jsonl", "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        if include_dev:
            train_data, temp_data = train_test_split(
                data, test_size=(1 - self.config.train_ratio), random_state=self.config.seed
            )
            dev_size = self.config.dev_ratio / (1 - self.config.train_ratio)
            dev_data, test_data = train_test_split(
                temp_data, test_size=(1 - dev_size), random_state=self.config.seed
            )
            splits = [("train", train_data), ("dev", dev_data), ("test", test_data)]
        else:
            train_data, test_data = train_test_split(
                data, test_size=(1 - self.config.train_ratio), random_state=self.config.seed
            )
            splits = [("train", train_data), ("test", test_data)]

        for name, split in splits:
            with open(self.processed_data_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for item in split:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")

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
            print(f"⚠️ JSON extraction failed: {e}")
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