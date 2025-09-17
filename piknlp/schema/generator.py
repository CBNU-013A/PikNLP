# piknlp/common/schema.py

from pydantic import BaseModel
from typing import Literal

class Review_Sentiment_Label(BaseModel):
    category: str
    review: Literal["pos", "neg", "none"]

class Review_Sentiment_Sample(BaseModel):
    sentence: str
    label: list[Review_Sentiment_Label]

class SentimentList(BaseModel):
    sentiments: dict[str, Literal["pos", "neg", "none"]]

class Review_Category_Sample(BaseModel):
    sentence: str
    label: dict[str, str]

class CategoryList(BaseModel):
    categories: dict[str, str]