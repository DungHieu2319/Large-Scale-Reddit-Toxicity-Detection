import os
import sys

# ── FIX CHO WINDOWS ───────────────────────────────────────────────────────────
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.makedirs("C:/Temp", exist_ok=True)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, size
from pyspark.ml.feature import Tokenizer
from pyspark.ml import PipelineModel
from pyspark.ml.clustering import BisectingKMeansModel

# ══════════════════════════════════════════════════════════════════════════════
# ⚠️  CHỈ CẦN SỬA 2 DÒNG NÀY
# ══════════════════════════════════════════════════════════════════════════════

# Đường dẫn tới processed_data của MÁY BẠN
MY_PROCESSED_DATA = "data/processed/processed_data"

# Tên output — mỗi người đặt tên khác nhau
# Người 2: "data/processed/features_person2"
# Người 3: "data/processed/features_person3"
MY_OUTPUT_PATH = "data/processed/features_person2"

# Đường dẫn tới model của NGƯỜI 1 gửi cho bạn
# Copy nguyên folder outputs/ từ máy người 1 vào máy bạn
PIPELINE_MODEL_PATH    = "outputs/pipeline_model"
KMEANS_MODEL_PATH      = "outputs/kmeans_model"
TOXIC_CLUSTER_ID_FILE  = "outputs/toxic_cluster_id.txt"

# ══════════════════════════════════════════════════════════════════════════════


def run():
    print("🚀 Starting Spark...")
    spark = SparkSession.builder \
        .appName("Transform - Person 2/3") \
        .master("local[*]") \
        .config("spark.driver.memory", "10g") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.sql.shuffle.partitions", "16") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.driver.extraJavaOptions",
                "-Djava.io.tmpdir=C:/Temp "
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                "--add-opens=java.base/java.lang=ALL-UNNAMED") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("C:/Temp/spark-checkpoint")

    # ── KIỂM TRA MODEL TỒN TẠI ───────────────────────────────────────────────
    if not os.path.exists(PIPELINE_MODEL_PATH):
        print(f"❌ Không tìm thấy pipeline model tại: {PIPELINE_MODEL_PATH}")
        print("   Hãy copy folder outputs/ từ máy người 1 vào máy bạn!")
        spark.stop()
        return

    if not os.path.exists(TOXIC_CLUSTER_ID_FILE):
        print(f"❌ Không tìm thấy toxic_cluster_id.txt tại: {TOXIC_CLUSTER_ID_FILE}")
        spark.stop()
        return

    # ── LOAD MODEL TỪ NGƯỜI 1 ────────────────────────────────────────────────
    print("📂 Loading pipeline + kmeans models từ người 1...")
    tfidf_model = PipelineModel.load(PIPELINE_MODEL_PATH)
    km_model    = BisectingKMeansModel.load(KMEANS_MODEL_PATH)

    with open(TOXIC_CLUSTER_ID_FILE, "r") as f:
        toxic_cluster_id = int(f.read().strip())

    print(f"   ✅ Toxic cluster ID: {toxic_cluster_id}")

    # ── LOAD DATA CỦA BẠN ────────────────────────────────────────────────────
    print(f"📂 Loading processed data từ: {MY_PROCESSED_DATA}")
    df = spark.read.parquet(MY_PROCESSED_DATA)
    df = df.dropna(subset=["clean_text"])
    df = df.repartition(16)

    # ── TOKENIZE ──────────────────────────────────────────────────────────────
    print("✂️ Tokenizing...")
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    df = tokenizer.transform(df)
    df = df.filter(size(col("words")) >= 3)
    df = df.checkpoint()

    total = df.count()
    print(f"   Total records: {total:,}")

    # ── TRANSFORM BẰNG MODEL CỦA NGƯỜI 1 ─────────────────────────────────────
    print("🔢 Transforming TF-IDF...")
    df_feat = tfidf_model.transform(df)
    df_feat = df_feat.checkpoint()

    print("🔵 Predicting clusters...")
    df_clustered = km_model.transform(df_feat)
    df_clustered = df_clustered.checkpoint()

    # ── GÁN LABEL ────────────────────────────────────────────────────────────
    print("🏷️ Labeling toxic/non-toxic...")
    df_labeled = df_clustered.withColumn(
        "label", when(col("cluster") == toxic_cluster_id, 1.0).otherwise(0.0)
    )

    # ── THỐNG KÊ ─────────────────────────────────────────────────────────────
    toxic    = df_labeled.filter(col("label") == 1.0).count()
    nontoxic = total - toxic
    print(f"\n📊 Kết quả:")
    print(f"   Total    : {total:,}")
    print(f"   Toxic    : {toxic:,} ({toxic/total*100:.1f}%)")
    print(f"   Non-toxic: {nontoxic:,} ({nontoxic/total*100:.1f}%)")

    # ── SAVE ──────────────────────────────────────────────────────────────────
    print(f"\n💾 Saving features to: {MY_OUTPUT_PATH}")
    df_labeled.select("clean_text", "label", "features") \
        .write.mode("overwrite").parquet(MY_OUTPUT_PATH)

    print(f"✅ DONE! Gửi folder '{MY_OUTPUT_PATH}' cho người 1 để train model.")
    spark.stop()


if __name__ == "__main__":
    run()