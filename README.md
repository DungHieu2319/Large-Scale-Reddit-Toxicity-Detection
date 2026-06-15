# Large-Scale Toxic Comment Detection in Reddit

## Giới thiệu (Introduction)
Dự án Khai phá dữ liệu lớn (Data Mining) nhằm phát hiện các bình luận độc hại (Toxicity Detection) trên nền tảng Reddit. Dự án kết hợp khả năng xử lý dữ liệu phân tán bằng **Apache Spark** và sức mạnh học sâu (Deep Learning) bằng **HuggingFace Transformers (BERT/DistilBERT)**.

## Kiến trúc Hệ thống (System Architecture)
Dự án được chia thành 2 luồng tiếp cận chính để so sánh hiệu năng:

### 1. Luồng Học máy truyền thống (Big Data Pipeline với PySpark)
- **Preprocessing (`src/preprocessing/preprocess.py`)**: Làm sạch văn bản, loại bỏ URL, ký tự đặc biệt, chữ thường, cắt từ (Tokenization) và loại bỏ từ dừng (StopWords) trên tập dữ liệu hàng chục triệu dòng.
- **Feature Engineering (`src/features/feature_engineering.py`)**: 
  - Sử dụng phương pháp **Rule-based hybrid labeling** dựa trên bộ từ khóa để gán nhãn dữ liệu một cách tự động (Nhanh và hiệu quả hơn thuật toán phân cụm KMeans).
  - Trích xuất đặc trưng văn bản bằng thuật toán **TF-IDF** (chiều dài vector 500).
- **Training (`src/models/train_model.py`)**: Đào tạo mô hình phân loại Logistic Regression sử dụng thư viện PySpark MLlib. Hỗ trợ Class Weighting để giải quyết vấn đề mất cân bằng dữ liệu.
- **Evaluation (`src/evaluation/evaluate.py`)**: Đánh giá hiệu năng mô hình trên tập Test bằng các chỉ số AUC, Precision, Recall, F1-score và Ma trận nhầm lẫn (Confusion Matrix).

### 2. Luồng Học Sâu (Deep Learning Pipeline với PyTorch)
- Sử dụng các mô hình ngôn ngữ lớn (LLMs) như **BERT** và **DistilBERT** để Fine-tune trực tiếp trên dữ liệu văn bản.
- Xử lý các mẫu mất cân bằng lớp (Class Imbalance) bằng **WeightedTrainer**.
- Tối ưu hóa huấn luyện trên nền tảng GPU (Hỗ trợ `bf16`, `tf32`, Gradient Checkpointing) qua các tệp:
  - `src/models/finetune_bert.py`
  - `src/models/train_distilbert_streaming.py`

## Cấu trúc thư mục (Directory Structure)
```text
├── data/                  # Thư mục chứa dữ liệu thô (raw) và đã xử lý (processed)
├── outputs/               # Chứa các mô hình đã được huấn luyện (Spark, PyTorch)
├── src/                   # Mã nguồn chính
│   ├── evaluation/        # Mã nguồn đánh giá (Metrics)
│   ├── features/          # Feature Engineering (TF-IDF, Embeddings)
│   ├── models/            # Mã nguồn huấn luyện (Spark LR, BERT)
│   └── preprocessing/     # Tiền xử lý dữ liệu văn bản
├── requirements.txt       # Các thư viện cần thiết
└── README.md              # Tài liệu dự án
```

## Cài đặt (Installation)
Yêu cầu hệ thống: **Python 3.9+**, **Java 8/11** (Bắt buộc để chạy PySpark).

```bash
# Tạo môi trường ảo
python -m venv .venv
# Kích hoạt trên Windows
.venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```
*(Trên Windows, Spark có thể yêu cầu tạo trước thư mục `D:/SparkTemp` hoặc cấu hình thêm biến môi trường `HADOOP_HOME`).*

## Hướng dẫn sử dụng (Usage)

### Cách 1: Chạy luồng PySpark (Học máy truyền thống)
Chạy tuần tự các tập lệnh sau từ thư mục gốc của dự án:
```bash
python src/preprocessing/preprocess.py
python src/features/feature_engineering.py
python src/models/train_model.py
python src/evaluation/evaluate.py
```

### Cách 2: Chạy luồng Deep Learning (BERT Fine-Tuning)
*(Khuyến nghị chạy trên máy tính có GPU hỗ trợ CUDA)*
```bash
python src/models/finetune_bert.py
```


