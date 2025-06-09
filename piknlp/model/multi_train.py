from piknlp.model.train import BaseTrainer
from piknlp.common.config import Config
from piknlp.schema.train import Input_Category_Example, Input_Category_Features
from transformers import ElectraModel
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
from transformers import ElectraModel
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler, SequentialSampler
import os
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Any
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
from sklearn.metrics import classification_report, f1_score
import numpy as np
from safetensors.torch import save_file
import json
from pathlib import Path

class MultiHeadClassifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = ElectraModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout_rate)

        # 카테고리마다 head를 따로 두는 구조
        self.heads = nn.ModuleDict({
            cat: nn.Linear(self.encoder.config.hidden_size, len(sub_labels))
            for cat, sub_labels in config.category.items()
        })

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = self.dropout(outputs.last_hidden_state[:, 0])  # [CLS] token

        logits = {
            cat: head(cls_output)
            for cat, head in self.heads.items()
        }
        return logits  # {"장소": logits1, "활동": logits2, ...}
    @classmethod
    def from_pretrained(cls, pretrained_dir: str | Path, config: Config):
        # 1) 모델 빈 껍데기 생성
        model = cls(config)
        # 2) safetensors 로드
        from safetensors.torch import load_file
        import os
        path = os.path.join(pretrained_dir, "model.safetensors")
        state_dict = load_file(path)
        # 3) 가중치 주입
        model.load_state_dict(state_dict)
        return model

class MultiHeadDictDataset(Dataset):
    def __init__(self, data: dict[str, torch.Tensor]):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.token_type_ids = data["token_type_ids"]
        self.labels = data["labels"]  # dict[str → Tensor]
        self.main_categories = list(self.labels.keys())
        self.length = self.input_ids.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels": {cat: self.labels[cat][idx] for cat in self.main_categories}
        }

def convert_examples_to_features_multihead(
    config: Config,
    examples: list[Input_Category_Example],
    tokenizer,
    max_length: int
) -> list[Input_Category_Features]:
    
    # 각 카테고리별로 {label_str: index} 매핑
    label_map: dict[str, dict[str, int]] = {
        cat: {label: idx for idx, label in enumerate(sub_labels)}
        for cat, sub_labels in config.category.items()
    }

    features: list[Input_Category_Features] = []

    for ex in examples:
        encoded = tokenizer(
            text=ex.sentence,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

        labels = {}
        for cat, label_str in ex.label.items():
            labels[cat] = label_map[cat].get(label_str, label_map[cat].get("none", 0))

        features.append(
            Input_Category_Features(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                token_type_ids=encoded.get("token_type_ids", [0] * max_length),
                labels=labels
            )
        )

    return features

class MultiHeadTrainer(BaseTrainer):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.main_categories = list(config.category.keys())
        self.sub_categories = {main: subs for main, subs in config.category.items()}

    def create_examples(self, lines: list[dict], mode: str) -> list[Input_Category_Example]:
        examples: list[Input_Category_Example] = []
        for idx, entry in enumerate(lines):
            sentence = entry["sentence"]
            raw = entry.get("label", {})

            # 1) 리스트면 category→subcat 맵으로 변환
            if isinstance(raw, list):
                label_map = {lbl["category"]: lbl["review"] for lbl in raw}
            # 2) 아니면 이미 메인→서브맵이라 가정
            else:
                label_map = raw

            # 3) main_categories 순서대로 꺼내서 없으면 "none"
            label: dict[str,str] = {
                cat: label_map.get(cat, "none")
                for cat in self.main_categories
            }

            guid = f"{mode}-{idx}"
            examples.append(Input_Category_Example(guid=guid, sentence=sentence, label=label))
        return examples
    
    def convert_examples_to_features(self, examples: list[Input_Category_Example]) -> list[Input_Category_Features]:
        return convert_examples_to_features_multihead(
            config=self.config, 
            examples=examples, 
            tokenizer=self.tokenizer, 
            max_length=self.config.max_seq_len
            )
    
    def convert_to_dataset(self, features: list[Input_Category_Features]) -> MultiHeadDictDataset:
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        self.logger.info(f"🔍 Category keys: {self.config.category.keys()}")
        # 카테고리별 label tensor
        label_tensors = {
            cat: torch.tensor([f.labels[cat] for f in features], dtype=torch.long)
            for cat in self.config.category.keys()
        }

        return MultiHeadDictDataset({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": label_tensors
        })
    
    
    def train(self):
        train_dataset = self.load_and_cache_examples("train")
        test_dataset = self.load_and_cache_examples("test")
        print(f"[DEBUG] train samples: {len(train_dataset)}")
        print(f"[DEBUG] test  samples: {len(test_dataset)}")
        for cat in train_dataset.main_categories:
            counts = torch.bincount(train_dataset.labels[cat])
            print(f"[DEBUG] train label distribution — {cat}: {counts.tolist()}")
        for cat in test_dataset.main_categories:
            counts = torch.bincount(test_dataset.labels[cat])
            print(f"[DEBUG] test label distribution — {cat}: {counts.tolist()}")

        train_dataloader = DataLoader(train_dataset, batch_size=self.config.train_batch_size, shuffle=True, num_workers=self.config.num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.eval_batch_size, shuffle=False, num_workers=self.config.num_workers)
        
        model = MultiHeadClassifier(self.config)
        model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * self.config.epochs)

        global_step = 0
        best_score = 0.0

        for epoch in range(self.config.epochs):
            print(f"\n🟢 Epoch {epoch + 1}/{self.config.epochs}")
            epoch_loss = 0.0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                transient=True
            ) as progress:
                task = progress.add_task(f"Training (epoch {epoch + 1})", total=len(train_dataloader))

                for batch in train_dataloader:
                    # Move inputs to device
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    token_type_ids = batch["token_type_ids"].to(self.device)
                    labels = {
                        cat: batch["labels"][cat].to(self.device)
                        for cat in self.main_categories
                    }

                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )

                    # Loss 계산
                    total_loss = sum(
                        F.cross_entropy(outputs[cat], labels[cat])
                        for cat in self.main_categories
                    )

                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    epoch_loss += total_loss.item()
                    global_step += 1

                    progress.update(task, advance=1)

            avg_loss = epoch_loss / len(train_dataloader)
            print(f"✅ Epoch {epoch + 1} 평균 Loss: {avg_loss:.4f}")

            eval_score = self._evaluate(model, test_dataloader)
            if eval_score > best_score:
                best_score = eval_score
                self._save_model(model, self.model_dir / "best_model")
                self.tokenizer.save_pretrained(self.model_dir / "best_model")
    
    def _save_model(self, model: MultiHeadClassifier, save_dir: str | None = None):
        if save_dir is None:
            save_dir = os.path.join(self.model_dir, "multihead")
        os.makedirs(save_dir, exist_ok=True)

        # 1. 모델 파라미터 저장 (.safetensors)
        model_path = os.path.join(save_dir, "model.safetensors")
        # 1) state_dict 가져오기
        state_dict = model.state_dict()
        # 2) 모든 텐서를 contiguous() 처리
        contig_state = {k: v.contiguous() for k, v in state_dict.items()}
        # 3) 저장
        save_file(contig_state, model_path)

        # 2. config 저장
        config_data = {
            "model_type": "multihead",
            "categories": self.main_categories,
            "category2label": self.config.category,
            "max_seq_length": self.config.max_seq_len,
        }
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        print(f"💾 모델 저장됨 (safetensors): {model_path}")

    @torch.no_grad()
    def _evaluate(self, model: MultiHeadClassifier, dataloader: DataLoader) -> float:
        model.eval()

        preds_dict: dict[str, list[int]] = {cat: [] for cat in self.main_categories}
        labels_dict: dict[str, list[int]] = {cat: [] for cat in self.main_categories}

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)

            labels = {
                cat: batch["labels"][cat].to(self.device)
                for cat in self.main_categories
            }

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )  # outputs: dict[str, Tensor]

            for cat in self.main_categories:
                logits = outputs[cat]  # shape: (batch, num_classes)
                preds = torch.argmax(logits, dim=-1)

                preds_dict[cat].extend(preds.cpu().tolist())
                labels_dict[cat].extend(labels[cat].cpu().tolist())

        # 전체 카테고리 평균 F1 계산
        f1_total = 0.0
        for cat in self.main_categories:
            f1 = f1_score(labels_dict[cat], preds_dict[cat], average="macro")
            print(f"[{cat}] F1 (macro): {f1:.4f}")
            f1_total += f1

        macro_f1 = f1_total / len(self.main_categories)
        print(f"[Overall] Macro F1: {macro_f1:.4f}")

        return macro_f1