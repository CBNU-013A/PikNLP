from pydantic import BaseModel
from typing import Literal

class CategoryList(BaseModel):
    categories: list[str]

class ReviewLabel(BaseModel):
    category: str
    review: Literal["pos", "neg", "none"]

class ReviewLabel_neu(BaseModel):
    category: str
    review: Literal["pos", "neg", "neu", "none"]

class ReviewSample(BaseModel):
    sentence: str
    label: list[ReviewLabel]

class ReviewSample_neu(BaseModel):
    sentence: str
    label: list[ReviewLabel_neu]

class SentimentList_neu(BaseModel):
    sentiments: dict[str, Literal["pos", "neg", "neu", "none"]]

class SentimentList(BaseModel):
    sentiments: dict[str, Literal["pos", "neg", "none"]]

class InputExample(BaseModel):
    """
    학습에 들어갈 데이터 클래스
    """
    guid: str
    sentence: str
    category: str
    sentiment: str

class InputFeatures(BaseModel):
    """
    Example 클래스를 토크나이징 후 모델에 넣을 수 있는 형태 클래스
    """
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    label: list[int]