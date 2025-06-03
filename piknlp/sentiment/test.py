# piknlp/sentiment/test.py

import logging
import torch

from piknlp.common.config import Config
from piknlp.common.logger import get_logger
from transformers import ElectraTokenizer, ElectraForSequenceClassification

from piknlp.schema.train import InputExample
from piknlp.sentiment.train import convert_examples_to_features

class BaseTester:
    logger: logging.Logger = get_logger(__name__)

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.config.cuda else "cpu")
        self.tokenizer = ElectraTokenizer.from_pretrained(self.config.model_name)
        self.model = ElectraForSequenceClassification.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.label_map = {i: lbl for i, lbl in enumerate(self.config.label_list)}

    def test(self) -> None:
        while True:
            sentence = input("입력 문장 (종료하려면 엔터): ").strip()
            if not sentence:
                break
            
            examples = [
                InputExample(
                    guid=f"test-{i}",
                    sentence=sentence,
                    category=cat,
                    label="none"
                )
                for i, cat in enumerate(self.config.category)
            ]
            
            features = convert_examples_to_features(
                self.config,
                examples,
                self.tokenizer,
                self.config.max_seq_len
            )
            for ex, feat in zip(examples, features):
                input_ids = torch.tensor([feat.input_ids], device=self.device)
                attention_mask = torch.tensor([feat.attention_mask], device=self.device)
                token_type_ids = torch.tensor([feat.token_type_ids], device=self.device)

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    pred = torch.argmax(outputs.logits, dim=1).item()
                    sentiment = self.label_map[pred]

                print(f"[{ex.category}] → {sentiment}")
                
class SentimentTester(BaseTester):
    def __init__(self, config: Config) -> None:
        super().__init__(config)