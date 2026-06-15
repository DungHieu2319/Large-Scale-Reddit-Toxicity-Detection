import os
import sys

os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.makedirs("D:/SparkTemp", exist_ok=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(PROJECT_ROOT)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.classification import LogisticRegression


def run_training():
    print("Starting Spark...")
    spark = SparkSession.builder \
        .appName("Toxic Detection - Training") \
        .master("local[4]") \
        .config("spark.driver.memory", "10g") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "4g") \
        .config("spark.local.dir", "D:/SparkTemp") \
        .config("spark.sql.shuffle.partitions", "400") \
        .config("spark.default.parallelism", "200") \
        .config("spark.sql.files.maxPartitionBytes", "64m") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "1024m") \
        .config("spark.io.compression.codec", "zstd") \
        .config("spark.sql.parquet.compression.codec", "zstd") \
        .config("spark.driver.extraJavaOptions",
                "-Djava.io.tmpdir=D:/SparkTemp "
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                "--add-opens=java.base/java.lang=ALL-UNNAMED") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("D:/SparkTemp/spark-checkpoint")

    print("Loading features...")

    FEATURE_PATHS = {
        "Feature 1": "data/processed/features",
        "Feature 2": "data/processed/features1",
        "Feature 3": "data/processed/features2", 
        "Feature 4": "data/processed/features3", 
    }

    dfs = []
    expected_dim = None

    def get_feature_dim(df):
        """Read vector dimension from Spark ML metadata if available."""
        try:
            metadata = df.schema["features"].metadata
            return metadata.get("ml_attr", {}).get("num_attrs")
        except Exception:
            return None

    for name, path in FEATURE_PATHS.items():
        if os.path.exists(path):
            tmp = spark.read.parquet(path).select("features", "label")
            dim = get_feature_dim(tmp)

            if expected_dim is None and dim is not None:
                expected_dim = dim

            if expected_dim is not None and dim is not None and dim != expected_dim:
                print(
                    f" {name}: sai số chiều features ({dim}), "
                    f"kỳ vọng {expected_dim}. Path: {path}"
                )
                print("      Hãy regenerate tập này với cùng numFeatures trước khi train.")
                spark.stop()
                return

            dim_info = f" | dim={dim}" if dim is not None else ""
            print(f"{name}: loaded{dim_info}")
            dfs.append(tmp)
        else:
            print(f" {name}: không tìm thấy {path} — bỏ qua")

    if not dfs:
        print("- Không có features nào!")
        spark.stop()
        return

    # gộp tất cả
    df = dfs[0]
    for d in dfs[1:]:
        df = df.unionByName(d)

    # Keep the source parquet partitions. The dataset is much larger than the
    # JVM heap, so coalescing it to 16 partitions and caching it can OOM while
    # Spark builds cached column batches.
    df = df.select("features", "label").dropna(subset=["features", "label"])

    # ── 2. THỐNG KÊ ───────────────────────────────────────────────────────────
    label_counts = {
        int(row["label"]): row["count"]
        for row in df.groupBy("label").count().collect()
    }
    toxic    = label_counts.get(1, 0)
    nontoxic = label_counts.get(0, 0)
    total    = toxic + nontoxic
    weight   = round(nontoxic / toxic, 2) if toxic > 0 else 1.0

    if total == 0:
        print("Dataset không có bản ghi hợp lệ để train.")
        spark.stop()
        return

    print(f"\nTổng data:")
    print(f"   Total    : {total:,}")
    print(f"   Toxic    : {toxic:,} ({toxic/total*100:.1f}%)")
    print(f"   Non-toxic: {nontoxic:,} ({nontoxic/total*100:.1f}%)")
    print(f"   Weight   : Toxic×{weight} | Non-toxic×1.0")


    df = df.withColumn(
        "classWeight",
        when(col("label") == 1.0, weight).otherwise(1.0)
    )


    print("\nTraining Logistic Regression trên toàn bộ data...")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        maxIter=10,
        regParam=0.01,
        elasticNetParam=0.0,
        aggregationDepth=4,
        maxBlockSizeInMB=16.0,
    )
    model = lr.fit(df)

    # ── 5. SAVE MODEL ─────────────────────────────────────────────────────────
    print("\nSaving model...")
    os.makedirs("outputs/model_tong", exist_ok=True)
    model.write().overwrite().save("outputs/model_tong/lr_toxic_model_tong")

    print("Training DONE! Model saved to outputs/model_tong/lr_toxic_model_tong")
    print("   Chạy tiếp: python src/evaluation/evaluate_model.py")
    spark.stop()


if __name__ == "__main__":
    run_training()