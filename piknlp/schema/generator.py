# piknlp/common/schema.py

from pydantic import BaseModel
from typing import Literal

class ReviewLabel(BaseModel):
    category: str
    review: Literal["pos", "neg", "none"]

class ReviewSample(BaseModel):
    sentence: str
    label: list[ReviewLabel]

class SentimentList(BaseModel):
    sentiments: dict[str, Literal["pos", "neg", "none"]]
