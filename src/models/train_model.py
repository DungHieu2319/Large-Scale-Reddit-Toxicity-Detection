import os
import sys

# ── FIX CHO WINDOWS ───────────────────────────────────────────────────────────
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.makedirs("C:/Temp", exist_ok=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(PROJECT_ROOT)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.classification import LogisticRegression


def run_training():
    print("🚀 Starting Spark...")
    spark = SparkSession.builder \
        .appName("Toxic Detection - Training") \
        .master("local[*]") \
        .config("spark.driver.memory", "10g") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.sql.shuffle.partitions", "16") \
        .config("spark.driver.extraJavaOptions",
                "-Djava.io.tmpdir=C:/Temp "
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                "--add-opens=java.base/java.lang=ALL-UNNAMED") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("C:/Temp/spark-checkpoint")

    # ── 1. LOAD FEATURES (PERSON 1 + 2) ──────────────────────────────────────
    print("📂 Loading features từ 2 người...")

    FEATURE_PATHS = {
        "Person 1": "data/processed/features",
        "Person 2": "data/processed/features2",
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
            tmp = spark.read.parquet(path)
            dim = get_feature_dim(tmp)

            if expected_dim is None and dim is not None:
                expected_dim = dim

            if expected_dim is not None and dim is not None and dim != expected_dim:
                print(
                    f"   ❌ {name}: sai số chiều features ({dim}), "
                    f"kỳ vọng {expected_dim}. Path: {path}"
                )
                print("      Hãy regenerate tập này với cùng numFeatures trước khi train.")
                spark.stop()
                return

            count = tmp.count()
            dim_info = f" | dim={dim}" if dim is not None else ""
            print(f"   ✅ {name}: {count:,} records{dim_info}")
            dfs.append(tmp)
        else:
            print(f"   ⚠️  {name}: không tìm thấy {path} — bỏ qua")

    if not dfs:
        print("❌ Không có features nào!")
        spark.stop()
        return

    # gộp tất cả
    df = dfs[0]
    for d in dfs[1:]:
        df = df.union(d)

    df = df.dropna(subset=["features", "label"])
    df = df.repartition(16)
    df = df.checkpoint()

    # ── 2. THỐNG KÊ ───────────────────────────────────────────────────────────
    total    = df.count()
    toxic    = df.filter(col("label") == 1.0).count()
    nontoxic = total - toxic
    weight   = round(nontoxic / toxic, 2) if toxic > 0 else 1.0

    print(f"\n📊 Tổng data:")
    print(f"   Total    : {total:,}")
    print(f"   Toxic    : {toxic:,} ({toxic/total*100:.1f}%)")
    print(f"   Non-toxic: {nontoxic:,} ({nontoxic/total*100:.1f}%)")
    print(f"   Weight   : Toxic×{weight} | Non-toxic×1.0")

    # ── 3. CLASS WEIGHT ───────────────────────────────────────────────────────
    df = df.withColumn(
        "classWeight",
        when(col("label") == 1.0, weight).otherwise(1.0)
    )

    # ── 4. TRAIN TOÀN BỘ ─────────────────────────────────────────────────────
    print("\n🤖 Training Logistic Regression trên toàn bộ data...")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        maxIter=10,
        regParam=0.01,
        elasticNetParam=0.0
    )
    model = lr.fit(df)

    # ── 5. SAVE MODEL ─────────────────────────────────────────────────────────
    print("\n💾 Saving model...")
    os.makedirs("outputs/model", exist_ok=True)
    model.write().overwrite().save("outputs/model/lr_toxic_model")

    print("✅ Training DONE! Model saved to outputs/model/lr_toxic_model")
    print("   Chạy tiếp: python src/evaluation/evaluate_model.py")
    spark.stop()


if __name__ == "__main__":
    run_training()