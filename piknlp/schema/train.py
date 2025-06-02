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