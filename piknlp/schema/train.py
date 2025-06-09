# piknlp/schema/train.py

from pydantic import BaseModel

class InputExample(BaseModel):
    guid: str
    sentence: str
    category: str
    label: str

class InputFeatures(BaseModel):
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    label: list[int]

class Input_Category_Example(BaseModel):
    guid: str
    sentence: str
    label: dict[str, str]

class Input_Category_Features(BaseModel):
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    labels: dict[str, int]  # {"장소": 2, "활동": 0, ...}