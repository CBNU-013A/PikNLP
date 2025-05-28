import os
import glob
import re

import torch
from torch.utils.data import DataLoader, SequentialSampler
from alive_progress import alive_bar
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, classification_report

from common.utils import MODEL_FOR_TOKEN_CLASSIFICATION, CONFIG_CLASSES, TOKENIZER_CLASS
from model.src.processor import Processor, load_and_cache_examples

class Evaluator:
    def __init__(self, args):
        self.args = args
    
    
    def compute_metrics(self, out_label_ids, preds):
        acc = accuracy_score(out_label_ids, preds)
        f1 = f1_score(out_label_ids, preds, average='macro')
        return {
            "accuracy": acc,
            "f1_macro": f1
        }
    
    def evaluate(self):
        processor = Processor(self.args)
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

        # GPU or CPU
        self.args.device = "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"

        # Data Preprocessing
        test_dataset = load_and_cache_examples(self.args, tokenizer, mode="test") if self.args.test_file else None

        results = {}
        checkpoint_dirs = glob.glob(os.path.join(self.args.output_dir, "checkpoint-*"))
        checkpoints = sorted(
            checkpoint_dirs,
            key=lambda x: int(re.findall(r"checkpoint-(\d+)", x)[0])
        )
        if not self.args.eval_all_ckpt:
            checkpoints = checkpoints[-1:]

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = MODEL_FOR_TOKEN_CLASSIFICATION[self.args.model_type].from_pretrained(
                checkpoint,
                config=config
            )
            model.to(self.args.device)
            result = self._evaluate(model, test_dataset, "test", global_step)
            result = dict((k + f"_{global_step}", v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            if len(checkpoints) > 1:
                for key in sorted(results.keys(), key=lambda k: (k.rsplit("_", 1)[0], int(k.rsplit("_", 1)[1]))):
                    f_w.write(f"{key}={str(results[key])}\n")
            else:
                for key in sorted(results.keys()):
                    f_w.write(f"{key}={str(results[key])}\n")

    def _evaluate(self, model, eval_dataset, mode, global_step=None, disable_bar=False):
        results = {}
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        with alive_bar(len(eval_dataloader), title=f'Evaluating {mode}', dual_line=True, disable=disable_bar) as bar:
            for batch in eval_dataloader:
                model.eval()
                batch = tuple(t.to(self.args.device) for t in batch)

                with torch.no_grad():
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[3],
                    }
                    if self.args.model_type != "distilbert":
                        inputs["token_type_ids"] = batch[2]
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
                bar()

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)

        # Detailed per-class metrics
        report = classification_report(out_label_ids, preds, target_names=self.args.sentiments, digits=3)
        # Write the report to a file
        output_dir = os.path.join(self.args.output_dir, mode)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        report_file = os.path.join(output_dir, f"{mode}_{global_step}_classification_report.txt" if global_step else f"{mode}_classification_report.txt")
        with open(report_file, "w") as f_report:
            f_report.write(report)

        result = self.compute_metrics(out_label_ids, preds)
        results.update(result)

        output_eval_file = os.path.join(output_dir, f"{mode}-{global_step}.txt" if global_step else f"{mode}.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(result.keys()):
                f_w.write(f"{key} = {result[key]}\n")
        return results