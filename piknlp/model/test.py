# piknlp/sentiment/test.py

import logging
import torch

from piknlp.common.config import Config
from piknlp.common.logger import get_logger
from transformers import ElectraTokenizer, ElectraForSequenceClassification

from piknlp.schema.train import InputExample, Input_Category_Example
from piknlp.model.train import convert_examples_to_features
from piknlp.model.multi_train import convert_examples_to_features_multihead, MultiHeadClassifier

class BaseTester:
    logger: logging.Logger = get_logger(__name__)

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.config.cuda else "cpu")
        self.tokenizer = ElectraTokenizer.from_pretrained(self.config.model_name)
        
        if self.config.model_type == "multihead":
            self.model = MultiHeadClassifier.from_pretrained(self.config.model_name, self.config) #TODO: 아마 안될꺼임
        else:
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

class CategoryTester(BaseTester):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        
    def test(self) -> None:
        while True:
            sentence = input("입력 문장 (종료하려면 엔터): ").strip()
            if not sentence:
                break
            
            example = Input_Category_Example(
                    guid=f"test-0",
                    sentence=sentence,
                    label={cat: "none" for cat in self.config.category.keys()}
                )
            examples = [example]
            features = convert_examples_to_features_multihead(
                config=self.config,
                examples=examples,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_len
            )
            features = features[0]
            input_ids = torch.tensor([features.input_ids], device=self.device)
            attention_mask = torch.tensor([features.attention_mask], device=self.device)
            token_type_ids = torch.tensor([features.token_type_ids], device=self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            for category, logits in outputs.items():
                pred_idx = torch.argmax(logits, dim=-1).item()
                pred_label = self.config.category[category][pred_idx]
                print(f"[{category}] → {pred_label}")