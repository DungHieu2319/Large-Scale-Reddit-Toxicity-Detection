"""
train_distilbert_streaming.py
===========================
Huấn luyện (Fine-tune) DistilBERT trực tiếp từ dữ liệu Parquet lớn (24GB)
sử dụng tính năng Streaming (IterableDataset) của HuggingFace.

Lợi ích:
- KHÔNG CẦN trích xuất features/embeddings ra ổ cứng trước.
- KHÔNG CẦN nạp toàn bộ 24GB vào RAM (tránh Out of Memory).
- Huấn luyện nhanh chóng trong thời gian cố định bằng cách giới hạn số steps (max_steps).

Yêu cầu:
- pip install transformers datasets torch evaluate scikit-learn
- Môi trường nên có GPU (CUDA) để huấn luyện nhanh chóng.
"""

import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from transformers.trainer_utils import get_last_checkpoint
import argparse

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions là logits (raw scores), chuyển sang xác suất (probabilities) để tính AUC
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()
    preds = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = 0.0
        
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc,
        "specificity": specificity
    }

def main():
    parser = argparse.ArgumentParser(description="Streaming Fine-tune DistilBERT on Large Data")
    parser.add_argument("--data_pattern", type=str, default="data/processed/features*",
                        help="Pattern của thư mục chứa dữ liệu ĐÃ CÓ NHÃN (VD: data/processed/features*)")
    parser.add_argument("--output_dir", type=str, default="outputs/distilbert_streaming_model",
                        help="Thư mục lưu mô hình sau khi huấn luyện")
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Lựa chọn 2: Train mẫu ngẫu nhiên trên 3.2 triệu comment (100,000 steps x 32 batch_size) để hoàn thành trong vài tiếng")
    parser.add_argument("--eval_steps", type=int, default=10000,
                        help="Số bước đánh giá/lưu model một lần (Nên để lớn để tránh lưu quá nhiều)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Kích thước batch (Train/Eval)")
    args = parser.parse_args()

    print("Bắt đầu quá trình Streaming Fine-tune (End-to-End) với DistilBERT")
    
    MODEL_NAME = "distilbert-base-uncased"
    print(f"Đang load Tokenizer và Model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Vì bài toán là nhị phân (Toxicity: 0 hoặc 1) nên num_labels=2
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # 1. Load DATA VÀO RAM VÀ CÂN BẰNG TỈ LỆ 50/50 (Sử dụng Memory-Mapping để không tràn RAM)
    from datasets import load_dataset, concatenate_datasets
    import os

    print(f"Đang nạp dữ liệu từ {args.data_pattern} (Chế độ Zero-RAM OOM Protection)...")
    
    # Đọc riêng biệt 2 thư mục để tránh lỗi Hive Partition
    dir_0 = os.path.join(args.data_pattern, "label=0.0")
    dir_1 = os.path.join(args.data_pattern, "label=1.0")
    
    ds_0 = load_dataset("parquet", data_dir=dir_0, split="train")
    ds_1 = load_dataset("parquet", data_dir=dir_1, split="train")
    
    # Gán lại cột label bị mất
    ds_0 = ds_0.add_column("label", [0] * len(ds_0))
    ds_1 = ds_1.add_column("label", [1] * len(ds_1))
    
    print("Đang tiến hành cân bằng dữ liệu (50% Toxic - 50% Non-Toxic)...")
    
    min_len = min(len(ds_0), len(ds_1))
    print(f"Tìm thấy {len(ds_0)} mẫu nhãn 0 và {len(ds_1)} mẫu nhãn 1.")
    
    if min_len == 0:
        raise ValueError("Lỗi: Một trong hai nhãn không có dữ liệu! Hãy kiểm tra lại thư mục trên Drive.")
        
    print(f"Sẽ cắt mỗi bên xuống còn {min_len} mẫu để đảm bảo cân bằng tuyệt đối!")
    
    ds_0 = ds_0.select(range(min_len))
    ds_1 = ds_1.select(range(min_len))
    
    # Ghép lại và Xóc đều (Global Shuffle)
    dataset = concatenate_datasets([ds_0, ds_1])
    dataset = dataset.shuffle(seed=42)
    print(f"Hoàn tất! Kích thước dữ liệu huấn luyện cuối cùng: {len(dataset)} dòng.")
    
    # 2. Tokenize dữ liệu (Chuyển text thành số để đưa vào mô hình)
    def tokenize_function(examples):
        tokenized = tokenizer(examples["clean_text"], truncation=True, max_length=128)
        tokenized["labels"] = examples["label"]
        return tokenized

    print("Đang thiết lập Pipeline Tokenize (On-the-fly)...")
    # Chúng ta xóa TOÀN BỘ các cột cũ (bao gồm cả cột 'features' của PySpark) để tránh lỗi Tensor collate
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    
    # 3. Chia tập Train và Validation
    # Dùng 2000 records đầu tiên làm tập Validation (Đánh giá)
    eval_dataset = tokenized_datasets.select(range(2000))
    # Phần còn lại làm tập Train
    train_dataset = tokenized_datasets.select(range(2000, len(tokenized_datasets)))
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 4. Cấu hình các tham số huấn luyện (Training Arguments)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.max_steps,              # QUAN TRỌNG: Dừng lại sau max_steps thay vì duyệt hết 24GB
        eval_strategy="steps",           # Đánh giá sau mỗi `eval_steps`
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),        # Bật FP16 nếu có GPU để train cực nhanh và tiết kiệm VRAM
        logging_steps=100,                     # In log thường xuyên để xem tiến độ
        remove_unused_columns=False,           # Đặt False để tránh lỗi Trainer xóa nhầm cột của IterableDataset
    )
    
    # 5. Khởi tạo Trainer
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer, # Cho transformers bản mới (>=4.46)
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,        # Cho transformers bản cũ
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    
    # 6. Bắt đầu huấn luyện!
    # Tự động tìm Checkpoint gần nhất để resume (nếu bị ngắt giữa chừng)
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            print(f"Tìm thấy bản lưu trước đó: {last_checkpoint}. Sẽ tiếp tục học (Resume) từ đây!")
            
    print(f"Đang tiến hành huấn luyện trong {args.max_steps} steps (Tương đương toàn bộ dữ liệu)...")
    print(f"   (Lưu ý: Bạn có thể nhấn Ctrl+C nếu muốn dừng sớm, model checkpoint gần nhất sẽ được lưu lại)")
    
    if last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    
    # 7. Đánh giá và lưu model
    results = trainer.evaluate()
    total_records = args.max_steps * args.batch_size
    
    print("\n" + "="*50)
    print(f"Model       : {args.output_dir}")
    print(f"Records     : {total_records:,} (max_steps={args.max_steps} x batch={args.batch_size})")
    print(f"AUC         : {results.get('eval_auc', 0.0):.4f}")
    print(f"Accuracy    : {results.get('eval_accuracy', 0.0):.4f}")
    print(f"Precision   : {results.get('eval_precision', 0.0):.4f}")
    print(f"Recall      : {results.get('eval_recall', 0.0):.4f}")
    print(f"F1-score    : {results.get('eval_f1', 0.0):.4f}")
    print(f"Specificity : {results.get('eval_specificity', 0.0):.4f}")
    print("="*50 + "\n")
    
    print(f"Đã hoàn tất và lưu mô hình tại {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
