# category/runner.py

from pathlib import Path
import pandas as pd
import json
from category.category_extractor import CategoryExtractor
from common.utils import INTERIM_CATEGORY_DIR, CATEGORY_PATH, RAW_DATA_DIR
from alive_progress import alive_bar

class CategoryExtractionRunner:
    def __init__(self, csv_path: str, max_reviews: int = 30):
        self.csv_path = Path(csv_path)
        self.max_reviews = max_reviews
        self.output_path = INTERIM_CATEGORY_DIR / f"{self.csv_path.stem}.categories.json"

    def run(self):
        reviews = self._load_and_sample_reviews()
        extractor = CategoryExtractor()
        categories = extractor.extract(reviews)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved to {self.output_path}, {len(categories)} categories found")

    def _load_and_sample_reviews(self) -> list[str]:
        df = pd.read_csv(self.csv_path)
        reviews = df["Review"].dropna()
        if len(reviews) <= self.max_reviews:
            return reviews.tolist()
        return reviews.sample(self.max_reviews, random_state=42).tolist()
    
    @staticmethod
    def merge_all_categories():
        all_files = INTERIM_CATEGORY_DIR.glob("*.categories.json")
        merged = set()
        for file in all_files:
            with open(file, encoding="utf-8") as f:
                merged.update(json.load(f))
        merged_list = sorted(merged)
        extractor = CategoryExtractor()
        merged_list = extractor.normalize(categories=merged_list)

        with open(CATEGORY_PATH, "w", encoding="utf-8") as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=2)
        print(f"✅ Merged {len(merged_list)} categories into {CATEGORY_PATH}")

    @classmethod
    def from_args(cls, args):
        if args.merge:
            cls.merge_all_categories()
        elif args.all:
            csv_files = list(RAW_DATA_DIR.glob("*.csv"))
            with alive_bar(len(csv_files), title="Processing CSV files", bar="filling", spinner="ball_belt") as bar:
                for csv_path in csv_files:
                    bar()
                    print(f"📄 Processing {csv_path.name}")
                    runner = cls(str(csv_path))
                    runner.run()
        elif args.csv:
            runner = cls(args.csv)
            runner.run()
        else:
            print("Error: CSV path required unless --merge is specified.")
    