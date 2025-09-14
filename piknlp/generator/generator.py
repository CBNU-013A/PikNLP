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
from piknlp.schema.generator import Review_Sentiment_Sample, Review_Category_Sample, SentimentList, CategoryList, Review_Sentiment_Label
from piknlp.generator.llm import call_ollama as call_llm
from piknlp.generator.llm import generate_dataset_prompt, generate_category_prompt, generate_dataset_batch_prompt, generate_category_batch_prompt

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
        labeled_data: list[Review_Sentiment_Sample | Review_Category_Sample] = []

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
                    try:
                        sample = f.result()
                        if sample.label:  # 유효한 레이블이 있는 경우만 추가
                            labeled_data.append(sample)
                    except Exception as e:
                        self.logger.error(f"Error processing review: {e}")
                    finally:
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
                # JSON 파싱 시도
                data = json.loads(json_str)
                # categories 또는 sentiments 키가 없으면 추가
                if "categories" not in data and "sentiments" not in data:
                    if self.config.task == "category":
                        data = {"categories": data}
                    elif self.config.task == "sentiment":
                        data = {"sentiments": data}
                return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"⚠️ JSON extraction failed: {e}")
        return "{}"

    def _parse_labels(self, response: SentimentList | CategoryList) -> dict[str, str] | list[Review_Sentiment_Label]:
        try:
            if isinstance(response, SentimentList):
                return [
                    Review_Sentiment_Label(category=cat, review=sentiment)
                    for cat, sentiment in response.sentiments.items()
                ]
            elif isinstance(response, CategoryList):
                return {cat: sub_cat for cat, sub_cat in response.categories.items()}
            else:   
                raise ValueError(f"Invalid response type: {type(response)}")
        except Exception as e:
            self.logger.error(f"Error parsing labels: {e}")
            return []
    
    def _process_review(self, review: str) -> Review_Sentiment_Sample | Review_Category_Sample:
        if self.config.task == "category":
            prompt = generate_category_prompt(review, self.config.category)
            response_class = CategoryList
        elif self.config.task == "sentiment":
            prompt = generate_dataset_prompt(review, self.config.category)
            response_class = SentimentList
        else:
            raise ValueError(f"Invalid task: {self.config.task}")

        raw_response = call_llm(prompt)
        clean_response = self._extract_json(raw_response)
        try:
            response_obj = response_class.model_validate_json(clean_response)
            labels = self._parse_labels(response_obj)
        except Exception as e:
            self.logger.error(f"Error parsing labels: {e}")
            self.logger.debug(f"Raw response: {raw_response}")
            self.logger.debug(f"Cleaned response: {clean_response}")
            labels = []

        if self.config.task == "sentiment":
            return Review_Sentiment_Sample(sentence=review, label=labels)
        elif self.config.task == "category":
            return Review_Category_Sample(sentence=review, label=labels)
        else:
            raise ValueError(f"Invalid task: {self.config.task}")