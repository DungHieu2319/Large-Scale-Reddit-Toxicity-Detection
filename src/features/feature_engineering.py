import os
import sys

# ── FIX CHO WINDOWS ───────────────────────────────────────────────────────────
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.makedirs("C:/Temp", exist_ok=True)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, array_contains, lit, avg, size
from pyspark.ml.feature import HashingTF, IDF, Normalizer, Tokenizer
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline

TOXIC_KEYWORDS = [
    "idiot", "stupid", "hate", "kill", "die", "racist", "moron",
    "dumb", "loser", "trash", "scum", "fool", "jerk", "bastard",
    "terrible", "awful", "disgusting", "pathetic", "worthless"
]


def run_feature_engineering():
    print("🚀 Starting Spark...")
    spark = SparkSession.builder \
        .appName("Feature Engineering - KMeans") \
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

    # ── 1. LOAD DATA ──────────────────────────────────────────────────────────
    print("📂 Loading processed data (PARQUET)...")
    df = spark.read.parquet("data/processed/processed_data")

    # ⚠️ SAMPLE TRƯỚC KHI LÀM BẤT CỨ GÌ KHÁC
    df = df.sample(fraction=0.05, seed=42)  # 5% ≈ 2.5 triệu records

    # drop dòng null
    df = df.dropna(subset=["clean_text"])

    # repartition sau khi sample
    df = df.repartition(16)

    print("Sample data:")
    df.select("clean_text").show(5, truncate=True)

    # ── 2. TOKENIZE ───────────────────────────────────────────────────────────
    print("✂️ Tokenizing...")
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    df = tokenizer.transform(df)

    # lọc câu quá ngắn
    df = df.filter(size(col("words")) >= 3)

    # checkpoint để cắt lineage, giải phóng bộ nhớ
    df = df.checkpoint()

    total = df.count()
    print(f"   Loaded {total:,} records")

    # ── 3. TF-IDF + NORMALIZE ─────────────────────────────────────────────────
    print("🔢 Building TF-IDF + L2 Normalize...")
    hashingTF  = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
    idf        = IDF(inputCol="rawFeatures", outputCol="tfidf", minDocFreq=5)
    normalizer = Normalizer(inputCol="tfidf", outputCol="features", p=2.0)

    pipeline = Pipeline(stages=[hashingTF, idf, normalizer])
    tfidf_model = pipeline.fit(df)
    df = tfidf_model.transform(df)

    # checkpoint lần 2 sau TF-IDF
    df = df.checkpoint()

    # ── 4. BISECTING K-MEANS ──────────────────────────────────────────────────
    print("🔵 Running BisectingKMeans (k=5)...")
    bkm = BisectingKMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=5,
        seed=42,
        maxIter=10,
        minDivisibleClusterSize=0.05
    )

    km_model     = bkm.fit(df)
    df_clustered = km_model.transform(df)
    df_clustered = df_clustered.checkpoint()

    silhouette = ClusteringEvaluator(
        featuresCol="features", predictionCol="cluster"
    ).evaluate(df_clustered)

    print(f"   Silhouette score: {silhouette:.4f}")

    print("   Cluster distribution:")
    df_clustered.groupBy("cluster").count().orderBy("cluster").show()

    # ── 5. XÁC ĐỊNH TOXIC ─────────────────────────────────────────────────────
    print("🔍 Identifying toxic cluster...")

    toxic_condition = lit(False)
    for kw in TOXIC_KEYWORDS:
        toxic_condition = toxic_condition | array_contains(col("words"), kw)

    df_clustered = df_clustered.withColumn(
        "has_toxic_kw", when(toxic_condition, 1.0).otherwise(0.0)
    )

    cluster_stats = df_clustered.groupBy("cluster").agg(
        avg("has_toxic_kw").alias("toxic_keyword_ratio")
    ).orderBy("cluster")

    cluster_stats.show()

    toxic_cluster_id = (
        cluster_stats
        .orderBy("toxic_keyword_ratio", ascending=False)
        .first()["cluster"]
    )

    print(f"   → Cluster {toxic_cluster_id} được xác định là TOXIC")

    df_labeled = df_clustered.withColumn(
        "label", when(col("cluster") == toxic_cluster_id, 1.0).otherwise(0.0)
    )

    # ── 6. THỐNG KÊ ───────────────────────────────────────────────────────────
    toxic_count    = df_labeled.filter(col("label") == 1.0).count()
    nontoxic_count = total - toxic_count
    toxic_pct      = toxic_count / total * 100

    print("📊 Kết quả label:")
    print(f"   Toxic     : {toxic_count:,} ({toxic_pct:.1f}%)")
    print(f"   Non-toxic : {nontoxic_count:,} ({100 - toxic_pct:.1f}%)")

    # ── 7. SAVE FEATURES ──────────────────────────────────────────────────────
    print("💾 Saving features...")
    df_labeled.select("clean_text", "label", "features") \
        .write.mode("overwrite").parquet("data/processed/features")

    # ── 8. SAVE PIPELINE + KMEANS MODEL (để người khác dùng) ─────────────────
    print("💾 Saving pipeline + kmeans models...")
    os.makedirs("outputs/pipeline_model", exist_ok=True)
    os.makedirs("outputs/kmeans_model", exist_ok=True)
    tfidf_model.write().overwrite().save("outputs/pipeline_model")
    km_model.write().overwrite().save("outputs/kmeans_model")
    with open("outputs/toxic_cluster_id.txt", "w") as f:
        f.write(str(toxic_cluster_id))

    print("✅ Feature engineering DONE!")
    spark.stop()


if __name__ == "__main__":
    run_feature_engineering()