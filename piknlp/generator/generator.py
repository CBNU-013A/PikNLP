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
        failed_reviews: list[str] = []  # 실패한 리뷰들을 수집

        # 리뷰를 5개씩 묶어서 배치 생성
        batch_size = 2
        review_batches = [reviews[i:i + batch_size] for i in range(0, len(reviews), batch_size)]
        self.logger.info(f"Created {len(review_batches)} batches")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[green]리뷰 배치 처리 중...", total=len(review_batches))
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                # 배치와 인덱스를 함께 전달
                futures = {executor.submit(self._process_review_batch, batch): i for i, batch in enumerate(review_batches)}
                for f in as_completed(futures):
                    batch_idx = futures[f]
                    try:
                        batch_samples = f.result()
                        for sample in batch_samples:
                            if sample.label:  # 유효한 레이블이 있는 경우만 추가
                                labeled_data.append(sample)
                    except Exception as e:
                        self.logger.error(f"Error processing review batch {batch_idx}: {e}")
                        # 실패한 배치의 리뷰들을 수집
                        failed_batch = review_batches[batch_idx]
                        failed_reviews.extend(failed_batch)
                    finally:
                        progress.update(task, advance=1)
        
        # 실패한 리뷰들을 개별 처리
        if failed_reviews:
            self.logger.info(f"Processing {len(failed_reviews)} failed reviews individually...")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("[yellow]실패한 리뷰 개별 처리 중...", total=len(failed_reviews))
                with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                    futures = [executor.submit(self._process_review, review) for review in failed_reviews]
                    for f in as_completed(futures):
                        try:
                            sample = f.result()
                            if sample.label:  # 유효한 레이블이 있는 경우만 추가
                                labeled_data.append(sample)
                        except Exception as e:
                            self.logger.error(f"Error processing individual review: {e}")
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
            # 여러 JSON 객체가 있을 수 있으므로 가장 큰 JSON 블록을 찾기
            matches = list(re.finditer(r"\{[\s\S]*\}", text))
            if not matches:
                self.logger.warning("No JSON pattern found in response")
                return "{}"
            
            # 가장 긴 JSON 블록 선택 (보통 가장 완전한 것)
            longest_match = max(matches, key=lambda m: len(m.group(0)))
            json_str = longest_match.group(0)
            
            # JSON 파싱 시도
            data = json.loads(json_str)
            
            # 배치 응답인지 확인 (results 키가 있는지)
            if "results" in data:
                return json_str  # 배치 응답은 그대로 반환
            
            # 단일 응답인 경우 기존 로직 적용
            if "categories" not in data and "sentiments" not in data:
                if self.config.task == "category":
                    data = {"categories": data}
                elif self.config.task == "sentiment":
                    data = {"sentiments": data}
            
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"⚠️ JSON extraction failed: {e}")
            self.logger.debug(f"Text that failed to parse: {text[:500]}...")
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
    
    def _validate_labels(self, labels: dict[str, str] | list[Review_Sentiment_Label]) -> bool:
        """생성된 레이블이 유효한지 검증"""
        try:
            if self.config.task == "category":
                if not isinstance(labels, dict):
                    return False
                
                # 각 카테고리별로 유효한 값인지 확인
                for category, value in labels.items():
                    if category not in self.config.category:
                        self.logger.warning(f"Unknown category: {category}")
                        return False
                    
                    if value not in self.config.category[category]:
                        self.logger.warning(f"Invalid value '{value}' for category '{category}'. Valid values: {self.config.category[category]}")
                        return False
                
                return True
                
            elif self.config.task == "sentiment":
                if not isinstance(labels, list):
                    return False
                
                # 감성 분석의 경우 'pos', 'neg', 'none'만 유효
                valid_sentiments = {'pos', 'neg', 'none'}
                for label in labels:
                    if not isinstance(label, Review_Sentiment_Label):
                        return False
                    if label.review not in valid_sentiments:
                        self.logger.warning(f"Invalid sentiment value: {label.review}. Valid values: {valid_sentiments}")
                        return False
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating labels: {e}")
            return False
    
    def _process_review_batch(self, reviews: list[str]) -> list[Review_Sentiment_Sample | Review_Category_Sample]:
        """리뷰 배치를 처리하여 개별 샘플들을 반환"""
        if self.config.task == "category":
            prompt = generate_category_batch_prompt(reviews, self.config.category)
        elif self.config.task == "sentiment":
            prompt = generate_dataset_batch_prompt(reviews, self.config.category)
        else:
            raise ValueError(f"Invalid task: {self.config.task}")

        raw_response = call_llm(prompt)
        self.logger.debug(f"Raw batch response: {raw_response}")
        clean_response = self._extract_json(raw_response)
        self.logger.debug(f"Cleaned batch response: {clean_response}")
        
        samples = []
        try:
            # 배치 응답 파싱
            if not clean_response or clean_response == "{}":
                self.logger.warning("Empty or invalid response, falling back to individual processing")
                raise ValueError("Empty response")
                
            batch_data = json.loads(clean_response)
            results = batch_data.get("results", [])
            self.logger.debug(f"Parsed results count: {len(results)}")
            
            if not results:
                self.logger.warning("No results in batch response, falling back to individual processing")
                raise ValueError("No results in response")
            
            for i, result in enumerate(results):
                if i >= len(reviews):
                    self.logger.warning(f"Result index {i} exceeds review count {len(reviews)}")
                    break
                    
                review_text = result.get("review", reviews[i])
                
                if self.config.task == "sentiment":
                    sentiments = result.get("sentiments", {})
                    if not sentiments:
                        self.logger.warning(f"No sentiments found for review {i}, skipping")
                        continue
                    labels = [
                        Review_Sentiment_Label(category=cat, review=sentiment)
                        for cat, sentiment in sentiments.items()
                    ]
                    # 레이블 검증
                    if not self._validate_labels(labels):
                        self.logger.warning(f"Invalid labels for review {i}, skipping")
                        continue
                    sample = Review_Sentiment_Sample(sentence=review_text, label=labels)
                elif self.config.task == "category":
                    categories = result.get("categories", {})
                    if not categories:
                        self.logger.warning(f"No categories found for review {i}, skipping")
                        continue
                    # 레이블 검증
                    if not self._validate_labels(categories):
                        self.logger.warning(f"Invalid labels for review {i}, skipping")
                        continue
                    sample = Review_Category_Sample(sentence=review_text, label=categories)
                else:
                    continue
                    
                samples.append(sample)
                
        except Exception as e:
            self.logger.error(f"Error parsing batch response: {e}")
            self.logger.debug(f"Raw response: {raw_response}")
            self.logger.debug(f"Cleaned response: {clean_response}")
            # 배치 처리 실패 시 예외를 다시 발생시켜서 상위에서 실패한 리뷰들을 수집하도록 함
            raise Exception(f"Batch processing failed: {e}")
        
        return samples

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
            
            # 레이블 검증
            if not self._validate_labels(labels):
                self.logger.warning(f"Invalid labels for review, returning empty labels")
                labels = [] if self.config.task == "sentiment" else {}
                
        except Exception as e:
            self.logger.error(f"Error parsing labels: {e}")
            self.logger.debug(f"Raw response: {raw_response}")
            self.logger.debug(f"Cleaned response: {clean_response}")
            labels = [] if self.config.task == "sentiment" else {}

        if self.config.task == "sentiment":
            return Review_Sentiment_Sample(sentence=review, label=labels)
        elif self.config.task == "category":
            return Review_Category_Sample(sentence=review, label=labels)
        else:
            raise ValueError(f"Invalid task: {self.config.task}")