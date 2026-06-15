"""
extract_embeddings.py (Tối ưu CPU - Phiên bản hoàn chỉnh)
==========================================================
Chiến lược 2 bước rõ rệt:
Bước 1: Spark chia nhỏ dữ liệu thành các file parquet riêng biệt -> TẮT SPARK.
Bước 2: PyTorch/ONNX đọc từng file, extract embeddings, lưu file mới -> DỌN RAM.

Các cải tiến so với bản gốc:
- Dùng ONNX Runtime thay PyTorch thuần (~2-3x nhanh hơn trên CPU)
- torch.inference_mode() thay no_grad()
- torch.set_num_threads() để dùng hết CPU cores
- torch.compile() cho PyTorch >= 2.0
- Tăng BATCH_SIZE lên 256
- Ghi DataFrame embeddings nhanh hơn (pd.concat 1 lần)
- Tự động fallback về PyTorch nếu ONNX chưa cài
"""

import os
import sys
import time
import glob
import gc
import logging

os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(PROJECT_ROOT)

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyspark.sql import SparkSession

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("extract_embeddings.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
BATCH_SIZE       = 64       # ONNX Runtime trên CPU cần batch nhỏ hơn để tránh OOM
MAX_LENGTH       = 64
MODEL_NAME       = "distilbert-base-uncased"
SPARK_PARTITIONS = 470      # Chia làm 470 phần (~100k records/phần)
USE_ONNX         = True     # Đặt False nếu không cài optimum[onnxruntime]

# Các thư mục
INPUT_FEATURES  = "data/processed/features"
TEMP_TEXT_DIR   = "data/processed/temp_text_partitions"
OUTPUT_EMB_DIR  = "data/processed/bert_embeddings"
FINAL_OUT_DIR   = "data/processed/bert_features"
ONNX_MODEL_DIR  = "data/processed/distilbert_onnx"   # Cache ONNX model

os.makedirs(TEMP_TEXT_DIR, exist_ok=True)
os.makedirs(OUTPUT_EMB_DIR, exist_ok=True)
os.makedirs(ONNX_MODEL_DIR, exist_ok=True)


# ── BƯỚC 1: DÙNG SPARK CHUẨN BỊ DỮ LIỆU ─────────────────────────────────────
def prepare_data_with_spark():
    """Dùng Spark chia nhỏ dữ liệu ra ổ cứng 1 lần duy nhất rồi tắt."""
    existing_files = glob.glob(f"{TEMP_TEXT_DIR}/*.parquet")
    if len(existing_files) > 0:
        log.info(f" Đã tìm thấy {len(existing_files)} file text tạm. Bỏ qua bước Spark.")
        return

    log.info(" Khởi động Spark để chia nhỏ dữ liệu (chỉ chạy 1 lần)...")
    spark = (
        SparkSession.builder
        .appName("Prepare Data for BERT")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.local.dir", "D:/SparkTemp")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    t0 = time.time()
    df = spark.read.parquet(INPUT_FEATURES).select("clean_text", "label")

    log.info(f"  Đang chia thành {SPARK_PARTITIONS} file nhỏ, vui lòng đợi...")
    df.repartition(SPARK_PARTITIONS).write.mode("overwrite").parquet(TEMP_TEXT_DIR)

    spark.stop()
    log.info(f" Xong bước chuẩn bị dữ liệu ({(time.time() - t0) / 60:.1f} phút). Đã tắt Spark.\n")


# ── BƯỚC 2A: LOAD MODEL (ONNX hoặc PyTorch) ──────────────────────────────────
def load_model():
    """
    Ưu tiên ONNX Runtime (nhanh hơn ~2-3x trên CPU).
    Fallback về PyTorch thuần nếu optimum chưa được cài.
    Cài ONNX: pip install optimum[onnxruntime]
    """
    # Dùng hết số CPU cores để tăng tốc tính toán
    torch.set_num_threads(os.cpu_count())
    log.info(f"   CPU threads : {os.cpu_count()}")

    if USE_ONNX:
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer

            log.info("   Backend     : ONNX Runtime ")

            # Export ONNX 1 lần, cache lại để lần sau dùng lại
            if not os.path.exists(os.path.join(ONNX_MODEL_DIR, "model.onnx")):
                log.info("  Export ONNX model lần đầu (chỉ chạy 1 lần)...")
                model = ORTModelForFeatureExtraction.from_pretrained(
                    MODEL_NAME, export=True
                )
                model.save_pretrained(ONNX_MODEL_DIR)
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                tokenizer.save_pretrained(ONNX_MODEL_DIR)
                log.info(f" Đã lưu ONNX model vào: {ONNX_MODEL_DIR}")
            else:
                log.info(f" Load ONNX model từ cache: {ONNX_MODEL_DIR}")
                model     = ORTModelForFeatureExtraction.from_pretrained(ONNX_MODEL_DIR)
                tokenizer = AutoTokenizer.from_pretrained(ONNX_MODEL_DIR)

            return tokenizer, model, "onnx"

        except ImportError:
            log.warning("  optimum[onnxruntime] chưa cài. Fallback về PyTorch.")
            log.warning("   Cài bằng: pip install optimum[onnxruntime]")

    # Fallback: PyTorch thuần
    from transformers import DistilBertTokenizer, DistilBertModel

    log.info("   Backend     : PyTorch (CPU)")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model     = DistilBertModel.from_pretrained(MODEL_NAME)
    model.eval()

    # torch.compile tăng ~20-30% trên PyTorch >= 2.0
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            log.info("   torch.compile: (PyTorch 2.0+)")
        except Exception as e:
            log.warning(f"   torch.compile:  Bỏ qua ({e})")

    return tokenizer, model, "pytorch"


# ── BƯỚC 2B: EXTRACT EMBEDDINGS (1 BATCH) ─────────────────────────────────────
def extract_batch(texts, tokenizer, model, backend: str) -> np.ndarray:
    """
    Trả về numpy array shape (n, 768) là CLS embeddings.
    Hỗ trợ cả ONNX và PyTorch backend.
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    if backend == "onnx":
        # ORTModel nhận dict tensors bình thường như PyTorch
        outputs = model(**encoded)
        res = outputs.last_hidden_state[:, 0, :].detach().numpy()
        del outputs, encoded
        return res
    else:
        with torch.inference_mode():   # Nhanh hơn no_grad ~5-10%
            outputs = model(**encoded)
            res = outputs.last_hidden_state[:, 0, :].numpy()
            del outputs, encoded
            return res


# ── BƯỚC 2C: XỬ LÝ TOÀN BỘ FILE ─────────────────────────────────────────────
def run_inference():
    """Quét các file đã chia, chạy model, quản lý RAM chặt chẽ."""
    log.info("Khởi động DistilBERT Inference (CPU Optimized)...")

    tokenizer, model, backend = load_model()
    log.info(f"   Backend đang dùng: {backend.upper()}\n")

    input_files = sorted(glob.glob(f"{TEMP_TEXT_DIR}/*.parquet"))
    total_files = len(input_files)

    if total_files == 0:
        log.error("Không tìm thấy file nào trong TEMP_TEXT_DIR. Hãy chạy bước Spark trước.")
        return

    log.info(f"Bắt đầu xử lý {total_files} files (BATCH_SIZE={BATCH_SIZE})...")
    t_start = time.time()
    processed_count = 0
    skipped_count   = 0
    file_times      = []  # Lưu thời gian xử lý từng file để tính ETA

    # tqdm cho vòng lặp file — hiển thị tiến độ tổng thể
    file_bar = tqdm(input_files, desc="Files", unit="file", dynamic_ncols=True)

    for i, file_path in enumerate(file_bar):
        base_name = os.path.basename(file_path)
        out_path  = os.path.join(OUTPUT_EMB_DIR, f"emb_{base_name}")

        # AUTO-RESUME: Bỏ qua file đã xử lý
        if os.path.exists(out_path):
            skipped_count += 1
            file_bar.set_postfix_str(f"skip={skipped_count}", refresh=False)
            continue

        t0 = time.time()

        try:
            # 1. Đọc file bằng Pandas
            df     = pd.read_parquet(file_path)
            texts  = df["clean_text"].fillna("").astype(str).tolist()
            labels = df["label"].tolist()
            n      = len(texts)
            n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

            # 2. Extract embedding từng batch — tqdm nội tuyến cho từng file
            all_embeddings = []
            batch_bar = tqdm(
                range(0, n, BATCH_SIZE),
                desc=f"Batch [{i+1}/{total_files}]",
                total=n_batches,
                unit="batch",
                leave=False,        # Tự xóa sau khi xong, không rác terminal
                dynamic_ncols=True,
            )
            for j in batch_bar:
                batch_embeds = extract_batch(
                    texts[j:j + BATCH_SIZE], tokenizer, model, backend
                )
                all_embeddings.append(batch_embeds)
                batch_bar.set_postfix(records=f"{min(j+BATCH_SIZE, n):,}/{n:,}")

            embeddings = np.vstack(all_embeddings)
            del all_embeddings  # Xóa sớm list các mảng con để trống RAM

            # 3. Ghi kết quả ra file Parquet
            # Tạo DataFrame một lần rồi insert cột label (tiết kiệm RAM gấp đôi so với pd.concat)
            df_out = pd.DataFrame(
                embeddings,
                columns=[f"emb_{k}" for k in range(embeddings.shape[1])]
            )
            df_out.insert(0, "label", labels)
            df_out.to_parquet(out_path, index=False, compression="snappy")

            elapsed = time.time() - t0
            file_times.append(elapsed)
            avg_time = sum(file_times) / len(file_times)
            remaining = total_files - (processed_count + skipped_count + 1)
            eta_min = (avg_time * remaining) / 60

            log.info(
                f"   ✓ [{i+1}/{total_files}] {n:,} records → {base_name} "
                f"({elapsed:.0f}s | ETA ~{eta_min:.0f} phút)"
            )
            processed_count += 1
            file_bar.set_postfix(done=processed_count, skip=skipped_count, ETA=f"{eta_min:.0f}m")

        except Exception as e:
            log.error(f"Lỗi file {base_name}: {e}")
            # Xóa file output lỗi nếu đã tạo dở
            if os.path.exists(out_path):
                os.remove(out_path)
            continue

        finally:
            # 4. Dọn dẹp RAM sau mỗi file (rất quan trọng trên CPU)
            # Python không cho phép dùng `del locals()[var]` để xóa biến cục bộ. Phải gán None.
            df = None
            texts = None
            labels = None
            all_embeddings = None
            embeddings = None
            df_out = None
            emb_df = None
            batch_embeds = None
            gc.collect()

    total_time = (time.time() - t_start) / 60
    log.info(f"\n{'='*60}")
    log.info(f"Hoàn thành! Đã xử lý: {processed_count} | Đã bỏ qua: {skipped_count} | Tổng: {total_files}")
    log.info(f"Tổng thời gian: {total_time:.1f} phút")
    log.info(f"{'='*60}\n")


# ── BƯỚC 3: GOM FILE LẠI THÀNH DỮ LIỆU CUỐI (TÙY CHỌN) ─────────────────────
def merge_final_data():
    """Sau khi chạy xong toàn bộ, dùng Spark gom lại 1 lần cuối."""
    emb_files = glob.glob(f"{OUTPUT_EMB_DIR}/*.parquet")
    if not emb_files:
        log.warning("Không có file embedding nào để gom. Bỏ qua bước merge.")
        return

    log.info(f"\nGom {len(emb_files)} file embeddings thành dataset cuối cùng...")
    spark = (
        SparkSession.builder
        .appName("Merge Embeddings")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    df_all = spark.read.parquet(OUTPUT_EMB_DIR)
    total  = df_all.count()

    df_all.write.mode("overwrite").partitionBy("label").parquet(FINAL_OUT_DIR)

    log.info(f"   ✓ Đã lưu {total:,} records vào: {FINAL_OUT_DIR}")
    spark.stop()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  DistilBERT Feature Extraction — CPU Optimized")
    log.info("=" * 60)

    prepare_data_with_spark()
    run_inference()
    merge_final_data()

    log.info("Pipeline hoàn tất!")