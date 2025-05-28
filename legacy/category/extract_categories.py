import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parent.parent))
from category.extractor_runner import CategoryExtractionRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find Topics from Reviews CSV")
    parser.add_argument("--csv", type=str, nargs="?", help="Full path to the CSV file")
    parser.add_argument("--merge", action="store_true", help="Merge all category jsons into category_set.json")
    parser.add_argument("--all", action="store_true", help="Run on all CSV files in the data/raw directory")
    args = parser.parse_args()
    CategoryExtractionRunner.from_args(args)