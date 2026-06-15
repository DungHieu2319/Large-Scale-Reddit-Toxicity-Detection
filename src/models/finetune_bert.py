"""
finetune_bert.py
===========================
Fine-tune BERT/DistilBERT End-to-End cho bai toan Toxicity Detection.

Su dung:
  python finetune_bert.py --model bert --sample 100000
  python finetune_bert.py --model bert  # toan bo data

Fix mat can bang class:
  - Balanced sampling: lay 50% toxic + 50% non-toxic tu toan bo dataset
  - WeightedTrainer voi CrossEntropyLoss co weight

Toi uu RTX 4060: bf16, tf32, gradient_checkpointing, pin_memory
"""

import os
import glob
import argparse

import torch
import evaluate
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIRS = [
    "data/processed/processed_data/processed_data_1",
    "data/processed/processed_data/processed_data_2",
    "data/processed/processed_data/processed_data_3",
    "data/processed/processed_data/processed_data_4",
]
DATA_FILES = {"train": [f"{d}/*.parquet" for d in DATA_DIRS]}


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return clf_metrics.compute(predictions=predictions, references=labels)


# ── Weighted Trainer ──────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    """
    Trainer voi weighted CrossEntropyLoss.
    Class toxic (hiem) duoc phat nang hon khi predict sai.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # tensor([w0, w1])

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )
        loss = loss_fct(
            outputs.logits.view(-1, model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


# ── Balanced sampling ─────────────────────────────────────────────────────────
def load_balanced_sample(n_per_class: int) -> Dataset:
    """
    Lay n_per_class dong toxic va n_per_class dong non-toxic tu 155M dong.
    Strategy:
      - Scan metadata de biet tong so dong (khong load data)
      - Lay proportional tu tung file parquet
      - Giu du ca 2 class, sau do can bang
    Ket qua: dataset 50/50 toxic vs non-toxic, thuc su ngau nhien.
    """
    all_files = []
    for pattern in DATA_FILES["train"]:
        all_files.extend(glob.glob(pattern))
    if not all_files:
        raise FileNotFoundError("Khong tim thay file parquet nao trong DATA_DIRS!")

    all_files = sorted(all_files)
    print(f"[BALANCED SAMPLE] Tim thay {len(all_files)} file parquet")
    print(f"Dang doc metadata...")

    file_sizes = {}
    total_rows = 0
    for f in all_files:
        nrows = pq.read_metadata(f).num_rows
        file_sizes[f] = nrows
        total_rows += nrows

    # Uoc tinh ti le toxic ~0.3% -> can doc nhieu hon de co du toxic
    # Doc ~10x so luong can thiet de dam bao lay du toxic
    n_total_needed = n_per_class * 2
    oversample_factor = max(1, int(1 / 0.003 * n_per_class / total_rows * 10))
    n_to_read = min(n_total_needed * oversample_factor, total_rows)

    print(f"Tong dataset: {total_rows:,} dong")
    print(f"Can: {n_per_class:,} toxic + {n_per_class:,} non-toxic = {n_total_needed:,} dong")
    print(f"Doc {n_to_read:,} dong ngau nhien de loc du toxic...")

    rng = np.random.default_rng(seed=42)
    toxic_dfs = []
    nontoxic_dfs = []
    n_toxic_collected = 0
    n_nontoxic_collected = 0

    for i, (fpath, nrows) in enumerate(file_sizes.items()):
        if n_toxic_collected >= n_per_class and n_nontoxic_collected >= n_per_class:
            break

        # So dong can doc tu file nay (proportional)
        k = min(round(n_to_read * nrows / total_rows), nrows)
        if k <= 0:
            continue

        df = pd.read_parquet(fpath, columns=["clean_text", "label"])
        df = df.sample(n=k, random_state=int(rng.integers(0, 99999)))

        # Tach theo class
        toxic = df[df["label"] == 1]
        nontoxic = df[df["label"] == 0]

        # Chi lay phan con thieu
        need_toxic = n_per_class - n_toxic_collected
        need_nontoxic = n_per_class - n_nontoxic_collected

        if need_toxic > 0 and len(toxic) > 0:
            take = min(len(toxic), need_toxic)
            toxic_dfs.append(toxic.iloc[:take])
            n_toxic_collected += take

        if need_nontoxic > 0 and len(nontoxic) > 0:
            take = min(len(nontoxic), need_nontoxic)
            nontoxic_dfs.append(nontoxic.iloc[:take])
            n_nontoxic_collected += take

    print(f"Da thu thap: {n_toxic_collected:,} toxic | {n_nontoxic_collected:,} non-toxic")

    if n_toxic_collected < n_per_class:
        print(f"[WARN] Chi lay duoc {n_toxic_collected:,} toxic (it hon {n_per_class:,})")

    combined = pd.concat(toxic_dfs + nontoxic_dfs, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    dataset = Dataset.from_pandas(combined, preserve_index=False)
    print(f"Dataset can bang: {len(dataset):,} dong | 50% toxic / 50% non-toxic")
    return dataset


def load_random_sample(n: int) -> Dataset:
    """True random sampling (khong can bang class)."""
    all_files = []
    for pattern in DATA_FILES["train"]:
        all_files.extend(glob.glob(pattern))
    all_files = sorted(all_files)

    print(f"[RANDOM SAMPLE] {len(all_files)} file parquet")
    file_sizes = {}
    total_rows = 0
    for f in all_files:
        nrows = pq.read_metadata(f).num_rows
        file_sizes[f] = nrows
        total_rows += nrows
    print(f"Tong: {total_rows:,} dong | Can lay: {n:,}")

    rng = np.random.default_rng(seed=42)
    dfs = []
    remaining = n
    for i, (fpath, nrows) in enumerate(file_sizes.items()):
        if remaining <= 0:
            break
        is_last = (i == len(file_sizes) - 1)
        k = remaining if is_last else min(round(n * nrows / total_rows), nrows, remaining)
        if k <= 0:
            continue
        df = pd.read_parquet(fpath, columns=["clean_text", "label"])
        dfs.append(df.sample(n=k, random_state=int(rng.integers(0, 99999))))
        remaining -= k

    combined = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)
    return Dataset.from_pandas(combined, preserve_index=False)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT/DistilBERT")
    parser.add_argument("--model", choices=["bert", "distilbert"], default="distilbert")
    parser.add_argument("--sample", type=int, default=None,
                        help="Tong so dong (vd: --sample 100000 -> 50k toxic + 50k non-toxic)")
    parser.add_argument("--balance", action="store_true",
                        help="Dung balanced sampling: 50%% toxic + 50%% non-toxic")
    args = parser.parse_args()

    print("=== Fine-tune BERT/DistilBERT ===")

    # 1. Chon model
    MODEL_NAME = "bert-base-uncased" if args.model == "bert" else "distilbert-base-uncased"
    output_dir = f"outputs/finetuned_{args.model}"
    if args.sample:
        mode = "balanced" if args.balance else "random"
        output_dir += f"_sample{args.sample // 1000}k_{mode}"

    print(f"Model   : {MODEL_NAME}")
    print(f"Output  : {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    model.gradient_checkpointing_enable()
    print("Gradient checkpointing: ON")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("TF32: ON")

    # 2. Load data
    if args.sample:
        n_per_class = args.sample // 2
        if args.balance:
            raw = load_balanced_sample(n_per_class)
        else:
            raw = load_random_sample(args.sample)
        dataset = raw.train_test_split(test_size=0.2, seed=42)
    else:
        print("Loading toan bo du lieu...")
        ds = hf_load_dataset("parquet", data_files=DATA_FILES)
        dataset = ds["train"].train_test_split(test_size=0.2, seed=42)

    train_labels = np.array(dataset["train"]["label"])
    n_total = len(train_labels)
    n_toxic = int(train_labels.sum())
    print(f"Train: {n_total:,} dong | toxic={n_toxic:,} ({n_toxic/n_total*100:.1f}%) | non-toxic={n_total-n_toxic:,} ({(n_total-n_toxic)/n_total*100:.1f}%)")

    # Class weights (can bang hon voi balanced sampling nen dung sqrt)
    w_nontoxic = n_total / (2 * (n_total - n_toxic)) if (n_total - n_toxic) > 0 else 1.0
    w_toxic    = n_total / (2 * n_toxic) if n_toxic > 0 else 1.0
    class_weights = torch.tensor([w_nontoxic, w_toxic], dtype=torch.float)
    print(f"Class weights: non-toxic={w_nontoxic:.3f} | toxic={w_toxic:.3f}")

    # 3. Tokenize
    def tokenize_fn(examples):
        result = tokenizer(examples["clean_text"], truncation=True, max_length=128)
        result["labels"] = examples["label"]
        return result

    print("Tokenizing...")
    cols_to_remove = [c for c in dataset["train"].column_names if c != "label"]
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    tokenized = tokenized.remove_columns(["label"])
    print(f"Columns: {tokenized['train'].column_names}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Training args
    is_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        fp16=False,
        bf16=is_cuda,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        logging_steps=100,
        logging_dir=f"{output_dir}/logs",
    )

    if is_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # 5. WeightedTrainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("torch.compile: SKIP (Windows khong ho tro Triton)")

    # 6. Train
    print("=== Bat dau huan luyen ===")
    trainer.train()

    # 7. Evaluate & Save
    print("=== Ket qua danh gia ===")
    print(trainer.evaluate())
    trainer.save_model(output_dir)
    print(f"Model da luu tai: {output_dir}")


if _name_ == "_main_":
    main()