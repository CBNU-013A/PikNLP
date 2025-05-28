from pathlib import Path
from transformers import ElectraTokenizer, AutoTokenizer
from transformers import ElectraForSequenceClassification, BertForSequenceClassification
from transformers import ElectraConfig, BertConfig

import random
import logging
import numpy as np
import torch

from seqeval import metrics as seqeval_metrics

INTERIM_CATEGORY_DIR = Path("data/interim/categories")
CATEGORY_PATH = Path("data/processed/category_set.json")

INTERIM_DATASET_DIR = Path("data/interim/dataset")
DATASET_PATH = Path("data/processed/labeled_reviews.jsonl")

RAW_DATA_DIR = Path("data/raw/")



# Training Arguments
CONFIG_CLASSES = {
    "koelectra-base": ElectraConfig,
    "koelectra-base-v2": ElectraConfig,
    "koelectra-base-v3": ElectraConfig,
    "koelectra-small": ElectraConfig,
    "koelectra-small-v3": ElectraConfig,
    "koelectra-small-v2": ElectraConfig,
    "kobert": BertConfig,
    "distilkobert": BertConfig,
    "kcelectra-base-22": ElectraConfig,
}

TOKENIZER_CLASS = {
    "koelectra-base": ElectraTokenizer,
    "koelectra-base-v2": ElectraTokenizer,
    "koelectra-base-v3": ElectraTokenizer,
    "koelectra-small": ElectraTokenizer,
    "koelectra-small-v2": ElectraTokenizer,
    "koelectra-small-v3": ElectraTokenizer,
    "kobert": AutoTokenizer,
    "distilkobert": AutoTokenizer,
    "kcelectra-base-22": ElectraTokenizer
}

MODEL_FOR_TOKEN_CLASSIFICATION = {
    "koelectra-base": ElectraForSequenceClassification,
    "koelectra-base-v2": ElectraForSequenceClassification,
    "koelectra-base-v3": ElectraForSequenceClassification,
    "koelectra-small-v3": ElectraForSequenceClassification,
    "koelectra-small": ElectraForSequenceClassification,
    "koelectra-small-v2": ElectraForSequenceClassification,
    "kobert": BertForSequenceClassification,
    "distilkobert": BertForSequenceClassification,
    "kcelectra-base-22": ElectraForSequenceClassification,
}

