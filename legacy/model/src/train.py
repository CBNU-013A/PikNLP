# model/train.py
# 모델 학습
import os
import json

import torch
from torch.utils.data import RandomSampler, DataLoader
from alive_progress import alive_bar
from attrdict import AttrDict

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from model.src.processor import Processor, load_and_cache_examples
from common.utils import (
    CONFIG_CLASSES,
    TOKENIZER_CLASS,
    MODEL_FOR_TOKEN_CLASSIFICATION
)
from model.src.evaluation import Evaluator

# K-Fold imports
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

class Trainer:
    def __init__(self, args):
        self.args = args

    def train(self):
        processor = Processor(self.args)
        # 1. 데이터 토크나이징까지 해서 모델 넣기 전까지 상태로 전처리
        # 2. 학습 루프 생성
        # 3. 평가
        # 4. 저장
        
        tokenizer = TOKENIZER_CLASS[self.args.model_type].from_pretrained(
            self.args.model_name_or_path,
            do_lower_case=self.args.do_lower_case
        )
        config = CONFIG_CLASSES[self.args.model_type].from_pretrained(
            self.args.model_name_or_path,
            num_labels=len(processor.categories),
            id2label={str(i): label for i, label in enumerate(processor.categories)},
            label2id={label: i for i, label in enumerate(processor.categories)},
        )
        model = MODEL_FOR_TOKEN_CLASSIFICATION[self.args.model_type].from_pretrained(
            self.args.model_name_or_path,
            config=config
        )

        # GPU or CPU
        self.args.device = "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"
        model.to(self.args.device)

        # Merge dev set into train set and drop dev entirely
        train_dataset_train = load_and_cache_examples(self.args, tokenizer, mode="train") if self.args.train_file else None
        dev_dataset = load_and_cache_examples(self.args, tokenizer, mode="dev") if self.args.dev_file else None
        from torch.utils.data import ConcatDataset
        if dev_dataset is not None:
            train_dataset = ConcatDataset([train_dataset_train, dev_dataset])
        else:
            train_dataset = train_dataset_train
        dev_dataset = None
        # always evaluate on test during training
        self.args.evaluate_test_during_training = True
        test_dataset = load_and_cache_examples(self.args, tokenizer, mode="test") if self.args.test_file else None

        # K-Fold cross validation on train_dataset if requested
        if hasattr(self.args, "kfold_num") and self.args.kfold_num > 1:
            all_labels = np.array([labels.item() for _, _, _, labels in train_dataset])
            # Group by sentence: assume examples are ordered by sentence across categories
            num_categories = len(processor.categories)
            groups = np.arange(len(train_dataset)) // num_categories
            kf = StratifiedGroupKFold(n_splits=self.args.kfold_num, shuffle=True, random_state=self.args.seed)
            for fold, (train_idx, val_idx) in enumerate(kf.split(all_labels, all_labels, groups=groups)):
                # Reinitialize model for each fold to avoid weight leakage
                model = MODEL_FOR_TOKEN_CLASSIFICATION[self.args.model_type].from_pretrained(
                    self.args.model_name_or_path,
                    config=config
                )
                model.to(self.args.device)
                print(f"Starting fold {fold + 1}/{self.args.kfold_num}")
                from torch.utils.data import Subset
                fold_train = Subset(train_dataset, train_idx)
                fold_dev = Subset(train_dataset, val_idx)
                # evaluate on dev fold
                self.args.evaluate_test_during_training = False

                # Create output directory for this fold
                original_output_dir = self.args.output_dir
                fold_output_dir = os.path.join(original_output_dir, f"fold_{fold+1}")
                os.makedirs(fold_output_dir, exist_ok=True)
                self.args.output_dir = fold_output_dir

                # Train and evaluate (step-wise results will be saved under fold_output_dir)
                self._train(model, fold_train, fold_dev, None)

                # Restore original output_dir
                self.args.output_dir = original_output_dir
            # After cross-validation, retrain on all data and evaluate on test set
            print("Re-training on full data and evaluating on test set")
            self.args.evaluate_test_during_training = True
            global_step, tr_loss = self._train(model, train_dataset, None, test_dataset)
        else:
            # No K-Fold, regular train/test
            global_step, tr_loss = self._train(model, train_dataset, None, test_dataset)

        print(f" global_step = {global_step}, average loss = {tr_loss}")

    def _train(self, 
               model, 
               train_dataset, 
               dev_dataset = None, 
               test_dataset = None):
        """
        균형 잡힌 클래스 분포를 가진 Train Loop
        """
        # 1. 데이터셋에서 각 감성(sentiment) 클래스 별로 개수 세기
        label_counts = {0: 0, 1: 0, 2: 0}  # {none: 0, neg: 0, neu: 0, pos: 0} 가정
        for _, _, _, labels in train_dataset:
            label = labels.item()  # 라벨 값 추출
            label_counts[label] += 1
        
        print(f"Label distribution: {label_counts}")
        
        # 2. 각 클래스에 가중치 부여 (적게 등장하는 클래스에 더 높은 가중치), 0 개 클래스는 가중치 0으로 설정
        max_count = max(label_counts.values()) if label_counts.values() else 0
        class_weights = {}
        for class_id, count in label_counts.items():
            # count가 0인 클래스는 가중치 0으로 처리
            class_weights[class_id] = (max_count / count) if count > 0 else 0.0

        # 3. 각 샘플에 해당 클래스의 가중치 할당
        weights = []
        for _, _, _, labels in train_dataset:
            label = labels.item()
            weights.append(class_weights.get(label, 0.0))
        
        # 4. WeightedRandomSampler 생성
        from torch.utils.data import WeightedRandomSampler
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True  # 중복 허용 (균형을 맞추기 위해)
        )
        
        train_dataloader = DataLoader(
            train_dataset, 
            sampler=train_sampler,
            batch_size=self.args.train_batch_size
        )
        
        # 이하 기존 코드와 동일
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        
        # Prepare Optimizer and Scheduler(linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * self.args.warmup_proportion), num_training_steps=t_total)

        if os.path.isfile(os.path.join(self.args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(self.args.model_name_or_path, "scheduler.pt")
        ):
            optimizer.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, "scheduler.pt")))
        
        # Train!
        global_step = 0
        tr_loss = 0.0

        model.zero_grad()
        for epoch in range(int(self.args.num_train_epochs)):
            print(f"Epoch {epoch+1} started.")
            with alive_bar(
                    len(train_dataloader), 
                    title=f'Epoch {epoch+1}',
                    bar='notes',
                    spinner='dots',
                    dual_line=True
                ) as bar:
                for step, batch in enumerate(train_dataloader):
                    model.train()
                    batch = tuple(t.to(self.args.device) for t in batch)
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[3]
                    }
                    if self.args.model_type != "distilbert":
                        inputs["token_type_ids"] = batch[2]
                    outputs = model(**inputs)
                    loss = outputs[0]
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                    loss.backward()
                    tr_loss += loss.item()
                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        len(train_dataloader) <= self.args.gradient_accumulation_steps and (step + 1) == len(train_dataloader)
                    ):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                        global_step += 1

                        if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                            if self.args.evaluate_test_during_training:
                                Evaluator(self.args)._evaluate(model, test_dataset, "test", global_step, disable_bar=True)
                            else:
                                Evaluator(self.args)._evaluate(model, dev_dataset, "dev", global_step, disable_bar=True)

                        if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                            output_dir = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(output_dir)
                            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                            print(f"Saving model checkpoint to {output_dir}")
                            if self.args.save_optimizer:
                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    bar()
            print(f"Epoch {epoch+1} done.")

        return global_step, tr_loss/global_step