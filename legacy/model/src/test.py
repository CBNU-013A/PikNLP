import torch
import glob
import re
import os
import json
from collections import defaultdict

from model.src.processor import Processor, convert_examples_to_features
from common.schema import InputExample
from common.utils import(
    CONFIG_CLASSES,
    TOKENIZER_CLASS,
    MODEL_FOR_TOKEN_CLASSIFICATION
)

class Tester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

        self.processor = Processor(args)
        self.categories = self.processor._get_category()
        self.label_map = {0: "pos", 1: "neg", 2: "neu", 3: "none"}

        # Find latest checkpoint
        checkpoint_dirs = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
        latest_ckpt = sorted(
            checkpoint_dirs, key=lambda x: int(re.findall(r"checkpoint-(\d+)", x)[0])
        )[-1] if checkpoint_dirs else args.output_dir

        self.tokenizer = TOKENIZER_CLASS[args.model_type].from_pretrained(args.model_name_or_path)
        self.model = MODEL_FOR_TOKEN_CLASSIFICATION[args.model_type].from_pretrained(latest_ckpt)
        self.model.to(self.device)
        self.model.eval()

    def test(self):
        while True:
            sentence = input("입력 문장 (종료하려면 엔터): ").strip()
            if not sentence:
                break

            examples = [
                InputExample(guid=f"test-{i}", sentence=sentence, category=cat, sentiment="none")
                for i, cat in enumerate(self.categories)
            ]

            features = convert_examples_to_features(self.args, examples, self.tokenizer, self.args.max_seq_len)
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
        
    def predict(self):
        data_dir = "data/raw"
        output_dir = "data/processed/predicts"
        os.makedirs(output_dir, exist_ok=True)
        all_file_results = {}
        files = os.listdir(data_dir)
        for file in files:
            if not file.endswith(".csv"):
                continue
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            file_result = defaultdict(lambda: defaultdict(int))
            for line in lines:
                line = line.strip()
                examples = [
                    InputExample(guid=f"{file}-{i}-{cat}", sentence=line, category=cat, sentiment="none")
                    for i, cat in enumerate(self.categories)
                ]

                features = convert_examples_to_features(self.args, examples, self.tokenizer, self.args.max_seq_len)
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
                        file_result[ex.category][sentiment] += 1
                        if sentiment in ["pos", "neg", "neu"]:
                            file_result[ex.category]["total"] += 1

            json_path = os.path.join(output_dir, f"{file}.summary.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(file_result, jf, ensure_ascii=False, indent=2)
            print(f"{file} 요약 저장 완료 → {json_path}")