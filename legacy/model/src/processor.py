import json
import os

import torch
from torch.utils.data import TensorDataset

from common.schema import InputExample, InputFeatures

class Processor:
    """
    데이터를 학습에 맞추어 클래스화
    """
    def __init__(self, args):
        self.args = args
        self.categories = self._get_category()
        self.sentiments = args.sentiments
    
    def _get_category(self) -> list:
        with open(self.args.category_dir, "r", encoding="utf-8") as f:
            category_set = json.load(f)
        return category_set

    def _read_file(self, input_file):
        data = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _create_examples(self, lines, mode):
        """
        문장들 토크나이징 전 클래스화
        멀티라벨 -> 단일라벨화
        """
        examples = []
        for idx, entry in enumerate(lines):
            sentence = entry["sentence"]
            label_dict = {lbl["category"]: lbl["review"] for lbl in entry["label"]}
            guid = f"{mode}-{idx}"

            for cat in self.categories:
                sentiment = label_dict.get(cat, "none")
                examples.append(
                    InputExample(
                        guid=guid, 
                        sentence=sentence, 
                        category=cat, 
                        sentiment=sentiment))
        return examples
    
    def get_examples(self, mode):
        """
        설정한 학습 데이터 종류 가져오기.
        이 함수를 호출하여 데이터 클래스화
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file
        
        return self._create_examples(
            self._read_file(os.path.join(self.args.data_dir, file_to_read)), mode
        )
    

def convert_examples_to_features(args, examples, tokenizer, max_length = 128):
    """
    example 데이터를 tokenizing 후 모델에 넣을 수 있는 형태로 변환
    """
    processor = Processor(args)
    label_map = {label: i for i, label in enumerate(processor.sentiments)}
    features = []

    for ex in examples:
        encoded = tokenizer(
            text = ex.sentence,
            text_pair = ex.category,
            truncation = True,
            max_length = max_length,
            padding = "max_length",
        )

        features.append(
            InputFeatures(
                input_ids = encoded["input_ids"],
                attention_mask = encoded["attention_mask"],
                token_type_ids = encoded.get("token_type_ids", [0] * max_length),
                label = [label_map[ex.sentiment]]
            )
        )
    return features

def load_and_cache_examples(args, tokenizer, mode):
    processor = Processor(args)
    cached_features_file = os.path.join(
        args.data_dir,
        f"cached_{list(filter(None, args.model_name_or_path.split('/'))).pop()}_{str(args.max_seq_len)}_{mode}"
    )
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file, weights_only=False)
    else:
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise ValueError(f"Invalid mode: {mode}")
        features = convert_examples_to_features(
            args, examples, tokenizer, args.max_seq_len)
        torch.save(features, cached_features_file)
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset