# piknlp/sentiment/train.py

import logging
import os
from typing import Any
import torch
from pathlib import Path
import json
import numpy as np


from piknlp.common.config import Config
from piknlp.common.logger import get_logger
from piknlp.schema.train import InputExample, InputFeatures

from sklearn.model_selection import StratifiedGroupKFold
from transformers import ElectraTokenizer, ElectraForSequenceClassification, ElectraConfig, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import ConcatDataset, Subset, TensorDataset, WeightedRandomSampler, DataLoader, SequentialSampler

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn, track
from sklearn.metrics import classification_report

class BaseTrainer:
    logger: logging.Logger = get_logger(__name__)
    
    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() and self.config.cuda else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.model_dir: Path = self.config.task_dir / "model" # /data/sentiment/model
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer: ElectraTokenizer = ElectraTokenizer.from_pretrained(self.config.model_name)

    def read_file(self, input_file: Path) -> list[dict]:
        """
        Read a file and return a list of dictionaries.
        Args:
            input_file: Path to the input file.
        Returns:
            A list of dictionaries.
        """
        data: list[dict] = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        self.logger.info(f"📂 Loaded {len(data)} samples from {input_file}")
        return data

    def create_examples(self, lines: list[dict], mode: str) -> list[InputExample]:
        """
        Create InputExample objects from a list of dictionaries.
        Args:
            lines: A list of dictionaries.
            mode: The mode of the dataset.
        Returns:
            A list of InputExample objects.
        """
        examples: list[InputExample] = []
        for idx, entry in enumerate(lines):
            sentence: str = entry["sentence"]
            label_dict: dict[str, str] = {lbl["category"]: lbl["review"] for lbl in entry["label"]}
            guid: str = f"{mode}-{idx}"

            for cat in self.config.category:
                label: str = label_dict.get(cat, "none")
                examples.append(InputExample(guid=guid, sentence=sentence, category=cat, label=label))
        return examples

    def convert_examples_to_features(self, examples: list[InputExample], tokenizer: ElectraTokenizer, max_length: int) -> list[InputFeatures]:
        """
        Convert a list of InputExample objects to a list of InputFeatures objects.
        Args:
            examples: A list of InputExample objects.
            tokenizer: A tokenizer object.
            max_length: The maximum length of the input sequence.
        Returns:
            A list of InputFeatures objects.
        """
        label_map: dict[str, int] = {label: i for i, label in enumerate(self.config.label_list)}
        features: list[InputFeatures] = []
        for ex in examples:
            encoded: dict[str, torch.Tensor] = tokenizer(
                text=ex.sentence,
                text_pair=ex.category,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            features.append(
                InputFeatures(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    token_type_ids=encoded.get("token_type_ids", [0] * max_length),
                    label=[label_map[ex.label]]
                )
            )
        return features

    def load_and_cache_examples(self, mode: str) -> ConcatDataset:
        """
        Load and cache examples from a file.
        Args:
            mode: The mode of the dataset.
        Returns:
            A ConcatDataset object.
        """
        
        # Define the path to the cached features file
        cached_features_dir: Path = self.model_dir / "interim" # /data/sentiment/model/interim
        cached_features_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 이름에서 실제 모델 이름만 추출 (monologg/koelectra-small-v3-discriminator -> koelectra-small-v3-discriminator)
        model_name = self.config.model_name.split("/")[-1] if "/" in self.config.model_name else self.config.model_name
        cached_features_file: Path = cached_features_dir / f"cached_{model_name}_{str(self.config.max_seq_len)}_{mode}.pt"
        
        # Check if the cached features file exists
        if cached_features_file.exists():
            # If the cached features file exists, load it
            features = torch.load(cached_features_file, weights_only=False)
            self.logger.info(f"✅ Loaded cached features from {cached_features_file}")
        else:
            # If the cached features file does not exist, create it and save it
            self.logger.info(f"🔄 Creating features for {mode} dataset")
            if mode == "train":
                examples: list[InputExample] = self.create_examples(self.read_file(self.config.task_dir / "dataset" / self.config.train_file), mode)
            elif mode == "test":
                examples: list[InputExample] = self.create_examples(self.read_file(self.config.task_dir / "dataset" / self.config.test_file), mode)
            else:
                raise ValueError(f"Invalid mode: {mode}")

            # Convert the examples to features
            features = self.convert_examples_to_features(examples, self.tokenizer, self.config.max_seq_len)

            # Save the features to the cached features file
            # TODO: check if the file is saved correctly
            torch.save(features, cached_features_file)
            self.logger.info(f"✅ Saved features to {cached_features_file}")
        
        # Create a TensorDataset from the features
        all_input_ids: torch.Tensor = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask: torch.Tensor = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids: torch.Tensor = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels: torch.Tensor = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset: TensorDataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset

    def train(self) -> None:
        """
        Train the model.
        """
        # Load the model
        config_class: ElectraConfig = ElectraConfig.from_pretrained(
            self.config.model_name,
            num_labels=len(self.config.category),
            id2label={str(i): label for i, label in enumerate(self.config.category)},
            label2id={label: i for i, label in enumerate(self.config.category)},
        )
        model: ElectraForSequenceClassification = ElectraForSequenceClassification.from_pretrained(
            self.config.model_name, config=config_class)
        model.to(self.device)
        self.logger.info(f"✅ Loaded model from {self.config.model_name}")

        # Load the train and test datasets
        train_dataset: ConcatDataset = self.load_and_cache_examples(mode="train")
        test_dataset: ConcatDataset = self.load_and_cache_examples(mode="test")
        self.logger.info(f"✅ Loaded train and test datasets")

        # If kfold_num is set, train the model with k-fold cross-validation
        if hasattr(self.config, "kfold_num") and self.config.kfold_num > 1:
            self.logger.info(f"K-Fold Training with {self.config.kfold_num} folds")
            
            all_labels: np.ndarray = np.array([labels.item() for _, _, _, labels in train_dataset])
            num_categories: int = len(self.config.category)
            groups: np.ndarray = np.arange(len(train_dataset)) // num_categories
            
            kf = StratifiedGroupKFold(n_splits=self.config.kfold_num, shuffle=True, random_state=self.config.kfold_seed)
            
            # Store results for each fold
            fold_results = []
            
            # Train the model with k-fold cross-validation
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                kfold_task = progress.add_task("[green]K-Fold Training...", total=self.config.kfold_num)
                # For Each Fold
                for fold, (train_idx, val_idx) in enumerate(kf.split(all_labels, all_labels, groups=groups)):
                    progress.update(kfold_task, completed=fold, description=f"[green]K-Fold Training...{fold+1}/{self.config.kfold_num}")

                    # Create a Subset of the train/dev dataset for the current fold
                    fold_train: Subset = Subset(train_dataset, train_idx)
                    fold_dev: Subset = Subset(train_dataset, val_idx)

                    # Define the output directory for the current fold
                    fold_output_dir: Path = self.model_dir / f"fold_{fold+1}" # /data/sentiment/model/fold_1

                    # Train the model on the current fold
                    step, loss = self._train(model = model,
                                train_dataset = fold_train,
                                dev_dataset = fold_dev,
                                test_dataset = test_dataset,
                                output_dir = fold_output_dir)
                    
                    # Evaluate and store results
                    fold_result = self.evaluate(model, fold_dev, "dev", fold+1)
                    fold_results.append(fold_result)
                    
                    self.logger.info(f"Fold {fold+1} done. global_step = {step}, average loss = {loss}")
                
                # Calculate and log average metrics across folds
                avg_metrics = {
                    "accuracy": np.mean([r["accuracy"] for r in fold_results]),
                    "f1": np.mean([r["f1"] for r in fold_results]),
                    "precision": np.mean([r["precision"] for r in fold_results]),
                    "recall": np.mean([r["recall"] for r in fold_results])
                }
                
                self.logger.info(f"Average metrics across {self.config.kfold_num} folds:")
                for metric, value in avg_metrics.items():
                    self.logger.info(f"{metric}: {value:.4f}")
                
                # Save average metrics
                metrics_file = self.model_dir / "kfold_average_metrics.txt"
                with open(metrics_file, "w") as f:
                    for metric, value in avg_metrics.items():
                        f.write(f"{metric}: {value:.4f}\n")
                
                # Re-train the model on the full data and evaluate on the test set
                self.logger.info("Re-training on full data and evaluating on test set")
                global_step, tr_loss = self._train(
                    model = model,
                    train_dataset = train_dataset,
                    dev_dataset = None,
                    test_dataset = test_dataset,
                    output_dir = self.model_dir / "final")
                self.logger.info("K-Fold Training done.")
        else:
            # No K-Fold, regular train/test
            self.logger.info("No K-Fold, regular train/test")
            # Train the model on the full data and evaluate on the test set
            global_step, tr_loss = self._train(model = model,
                                                train_dataset = train_dataset,
                                                dev_dataset = None,
                                                test_dataset = test_dataset,
                                                output_dir = self.model_dir / "final")
            self.logger.info(f"global_step = {global_step}, average loss = {tr_loss}")
            self.logger.info("Training done.")

    def _train(self, 
               model: ElectraForSequenceClassification, 
               train_dataset: ConcatDataset, 
               dev_dataset: ConcatDataset | None, 
               test_dataset: ConcatDataset | None,
               output_dir: Path) -> tuple[int, float]:
        """
        Train the model.
        Args:
            model: A model object.
            train_dataset: A train dataset object.
            dev_dataset: A dev dataset object.
            test_dataset: A test dataset object.
            output_dir: The directory to save the model.
        Returns:
            A tuple of the global step and the average loss.
        """
        self.logger.info(f"🚀 Training Started...")
        
        # Early stopping setup
        best_f1 = 0.0
        patience = self.config.early_stopping_patience
        patience_counter = 0
        best_model_path = None
        
        # 1. Count the number of samples for each label
        label_counts: dict[int, int] = {label: 0 for label in self.config.label_list}
        for _, _, _, labels in train_dataset:
            label_counts[labels.item()] += 1
        self.logger.debug(f"Label distribution: {label_counts}")
        
        # 2. Calculate the class weights(less frequent class has higher weight)
        max_count: int = max(label_counts.values()) if label_counts.values() else 0
        class_weights: dict[int, float] = {}
        for class_id, count in label_counts.items():
            # class with 0 count has weight 0
            class_weights[class_id] = (max_count / count) if count > 0 else 0.0
        
        # 3. Assign weights to each sample
        weights: list[float] = [class_weights.get(labels.item(), 0.0) for _, _, _, labels in train_dataset]
        self.logger.debug(f"Class weights: {class_weights}")

        # 4. Create a WeightedRandomSampler
        self.logger.debug(f"Weights: {weights}")
        train_sampler: WeightedRandomSampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True # allow sampling of the same sample multiple times
        )
        self.logger.debug(f"Train sampler: {train_sampler}")

        # 5. Create a DataLoader
        train_dataloader: DataLoader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.config.train_batch_size
        )
        self.logger.debug(f"Train dataloader: {train_dataloader}")

        # 6. Calculate epochs if max_steps is set
        if self.config.max_steps > 0:
            t_total = self.config.max_steps
            self.config.epochs = self.config.max_steps // (len(train_dataloader) // self.config.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.config.gradient_accumulation_steps * self.config.epochs
        
        # 7. Prepare Optimizer and Scheduler(linear warmup and decay)
        no_decay: list[str] = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters: list[dict[str, Any]] = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.config.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        # 7-1. Optimizer : AdamW, lr: default 5e-5, eps: default 1e-8
        optimizer: AdamW = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        # 7-2. Scheduler : linear warmup and decay, warmup_proportion: default 0.1
        scheduler: get_linear_schedule_with_warmup = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(t_total * self.config.warmup_proportion),
            num_training_steps=t_total
        )
        
        # 8. Load optimizer and Scheduler if they exist
        if os.path.isfile(os.path.join(self.config.model_name, "optimizer.pt")) and os.path.isfile(
            os.path.join(self.config.model_name, "scheduler.pt")
        ):
            optimizer.load_state_dict(torch.load(os.path.join(self.config.model_name, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.config.model_name, "scheduler.pt")))

        # 9. Training Loop
        global_step: int = 0
        tr_loss: float = 0.0
        model.zero_grad()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            # 9-1. Epoch Loop
            for epoch in range(int(self.config.epochs)):
                # 9-1-1. Epoch Start
                self.logger.info(f"Epoch {epoch+1} started.")
                task = progress.add_task(f"[green]Epoch {epoch+1}", total=len(train_dataloader))

                # 9-1-2. Step Loop
                for step, batch in track(enumerate(train_dataloader), description="[green]Training...", total=len(train_dataloader)):
                    # 9-1-2-1. Step Start
                    model.train()
                    # 9-1-2-2. Batch to Device
                    batch = tuple(t.to(self.device) for t in batch)
                    # 9-1-2-3. Forward Pass
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "labels": batch[3]
                    }
                    outputs = model(**inputs)
                    # 9-1-2-4. Calculate Loss
                    loss = outputs[0]
                    # 9-1-2-5. Gradient Accumulation
                    if self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps
                    # 9-1-2-6. Backward Pass
                    loss.backward()
                    # 9-1-2-7. Update Loss
                    tr_loss += loss.item()
                    # 9-1-2-8. Gradient Clipping & Optimizer Step
                    if (step + 1) % self.config.gradient_accumulation_steps == 0 or (
                        len(train_dataloader) <= self.config.gradient_accumulation_steps and (step + 1) == len(train_dataloader)
                    ):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                        global_step += 1
                    
                    progress.update(task, completed=step+1)
                
                # 9-1-3. Epoch End - Evaluate & Save
                self.logger.info(f"Epoch {epoch+1} finished. Evaluating...")
                
                # Evaluate on dev set if available
                if dev_dataset is not None:
                    dev_result = self.evaluate(model, dev_dataset, "dev", epoch+1)
                    current_f1 = dev_result["f1"]
                    
                    # Early stopping check
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        patience_counter = 0
                        # Save best model
                        best_model_path = output_dir / f"best_model"
                        best_model_path.mkdir(parents=True, exist_ok=True)
                        self.save_best_model(model=model, output_dir=best_model_path)
                        self.logger.info(f"✅ New best model saved to {best_model_path} with F1: {best_f1:.4f}")
                    else:
                        patience_counter += 1
                        self.logger.info(f"Early stopping patience: {patience_counter}/{patience}")
                        
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                
                # Evaluate on test set
                self.evaluate(model, test_dataset, "test", epoch+1)
                
                # Save checkpoint at end of each epoch
                epoch_output_dir = output_dir / f"ckpt" /f"epoch_{epoch+1}"
                epoch_output_dir.mkdir(parents=True, exist_ok=True)
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(epoch_output_dir)
                self.logger.info(f"✅ Model saved to {epoch_output_dir}")
                
                self.logger.info(f"Epoch {epoch+1} done.")

        return global_step, tr_loss

    def evaluate(self, 
                 model: ElectraForSequenceClassification, 
                 eval_dataset: ConcatDataset, 
                 mode: str, 
                 epoch: int = -1) -> dict:
        
        eval_sampler: SequentialSampler = SequentialSampler(eval_dataset)
        eval_dataloader: DataLoader = DataLoader(eval_dataset, 
                                                 sampler=eval_sampler, 
                                                 batch_size=self.config.eval_batch_size,
                                                 num_workers=self.config.eval_num_workers)

        model.eval()
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in track(eval_dataloader, description=f"[green]Evaluating {mode}...", total=len(eval_dataloader)):
            model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)


        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)

        # Detailed per-class metrics
        report = classification_report(out_label_ids, preds, target_names=self.config.category, digits=3)
        
        # Create evaluation output directory  
        eval_output_dir = self.model_dir / "evaluation" / mode
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write the report to a file
        report_file = eval_output_dir / f"{mode}_epoch_{epoch}_classification_report.txt"
        with open(report_file, "w") as f_report:
            f_report.write(report)

        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        accuracy = accuracy_score(out_label_ids, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(out_label_ids, preds, average='weighted')
        
        result = {
            "eval_loss": eval_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        # Write the result to a file
        result_file = eval_output_dir / f"{mode}_epoch_{epoch}_results.txt"
        with open(result_file, "w") as f_w:
            for key in sorted(result.keys()):
                f_w.write(f"{key} = {result[key]}\n")

        self.logger.info(f"📊 {mode.upper()} Results - Epoch {epoch}: Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Loss: {eval_loss:.4f}")
        
        return result

    def upload_to_huggingface(self, model_path: Path, repo_name: str, token: str) -> None:
        """
        학습된 모델을 Huggingface Hub에 업로드합니다.
        Args:
            model_path: 업로드할 모델의 경로
            repo_name: Huggingface Hub의 저장소 이름 (예: "username/model-name")
            token: Huggingface API 토큰
        """
        from huggingface_hub import HfApi, create_repo
        
        self.logger.info(f"🚀 Uploading model to Huggingface Hub: {repo_name}")
        
        try:
            # 저장소가 없으면 생성
            create_repo(repo_name, token=token, exist_ok=True)
            
            # API 클라이언트 생성
            api = HfApi()
            
            # 모델 파일 업로드
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_name,
                token=token
            )
            
            self.logger.info(f"✅ Model successfully uploaded to {repo_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to upload model: {str(e)}")
            raise

    def save_best_model(self, model: ElectraForSequenceClassification, output_dir: Path) -> None:
        """
        최적의 모델을 저장합니다.
        Args:
            model: 저장할 모델
            output_dir: 저장할 디렉토리
        """
        # 모델 저장
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # README.md 생성
        readme_content = f"""

# {self.config.task} Model

## Model Details
- Task: {self.config.task}
- Base Model: {self.config.model_name}
- Categories: {', '.join(self.config.category)}
- Labels: {', '.join(self.config.label_list)}

## Training Details
- Max Sequence Length: {self.config.max_seq_len}
- Batch Size: {self.config.train_batch_size}
- Learning Rate: {self.config.learning_rate}
- Epochs: {self.config.epochs}
- Early Stopping Patience: {self.config.early_stopping_patience}
"""
        
        with open(output_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        self.logger.info(f"✅ Model saved to {output_dir}")

class SentimentTrainer(BaseTrainer):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

class CategoryTrainer(BaseTrainer):
    def __init__(self, config: Config) -> None:
        super().__init__(config)