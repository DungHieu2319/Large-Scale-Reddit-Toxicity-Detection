import os
import sys
import time

# ── FIX CHO WINDOWS ───────────────────────────────────────────────────────────
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.makedirs("D:/SparkTemp", exist_ok=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(PROJECT_ROOT)

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, size, avg, count
)
from pyspark.ml.feature import HashingTF, IDF, Normalizer, Tokenizer
from pyspark.ml import Pipeline

TOXIC_KEYWORDS = [
    # Insults
    "idiot", "idiots", "stupid", "moron", "morons", "dumb", "dumbass",
    "fool", "fools", "retard", "retarded", "imbecile", "ignorant",
    "brainless", "dimwit", "halfwit", "clueless", "mindless", "braindead",
    # Hate
    "hate", "hater", "racist", "racism", "sexist", "sexism",
    "bigot", "bigotry", "homophobic", "transphobic", "nazi", "fascist",
    # Violence
    "kill", "murder", "shoot", "stab", "attack", "destroy",
    "hurt", "harm", "beat", "rape", "assault", "threat", "threaten",
    "die", "dies", "dying",
    # Xúc phạm
    "scum", "trash", "garbage", "disgusting", "pathetic", "worthless",
    "useless", "loser", "losers", "failure", "freak", "terrible",
    "awful", "horrible", "despicable", "vile", "wretched",
    # Profanity
    "jerk", "bastard", "asshole", "bitch", "cunt", "scumbag",
]

TOXIC_PHRASE_PATTERN = "|".join([
    r"kill\s+your\s*self", r"go\s+die",
    r"you\s+are\s+(a\s+)?(piece\s+of\s+)?trash",
    r"you\s+are\s+(a\s+)?loser",
    r"you\s+(should|deserve)\s+to\s+die",
    r"nobody\s+(likes|wants|cares)\s+(about\s+)?you",
    r"waste\s+of\s+(space|time|oxygen|life)",
    r"you\s+are\s+(so\s+)?(stupid|dumb|pathetic|worthless)",
    r"shut\s+up", r"get\s+lost", r"go\s+away",
    r"you\s+dont\s+belong", r"not\s+welcome\s+here",
    r"go\s+to\s+hell", r"drop\s+dead",
    r"nobody\s+asked\s+you", r"no\s+one\s+cares",
])

# Một regex duy nhất — tránh 80+ OR conditions trong query plan
KEYWORD_PATTERN        = r"\b(" + "|".join(TOXIC_KEYWORDS) + r")\b"
COMBINED_TOXIC_PATTERN = KEYWORD_PATTERN + "|" + TOXIC_PHRASE_PATTERN


def create_spark_session():
    return (
        SparkSession.builder
        .appName("Feature Engineering v5 - Rule-Based No KMeans")
        .master("local[4]")
        # Keep enough headroom for Windows, Python, and JVM native memory
        # on a 16 GB machine.
        .config("spark.driver.memory",         "6g")
        .config("spark.driver.maxResultSize",  "1g")
        .config("spark.memory.offHeap.enabled", "false")
        .config("spark.memory.storageFraction", "0.2")
        .config("spark.local.dir", "D:/SparkTemp")
        .config("spark.sql.files.maxPartitionBytes",             "64m")
        .config("spark.sql.shuffle.partitions",                   "256")
        .config("spark.sql.adaptive.enabled",                      "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled",   "true")
        .config("spark.sql.adaptive.skewJoin.enabled",             "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64m")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "128m")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.driver.extraJavaOptions",
                "-Djava.io.tmpdir=D:/SparkTemp "
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                "--add-opens=java.base/java.lang=ALL-UNNAMED")
        .getOrCreate()
    )


def run_feature_engineering():
    t_start = time.time()
    print("Starting Spark (v5 - Rule-Based, No KMeans, No Checkpoint)...")
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    # ── 1. LOAD DATA ──────────────────────────────────────────────────────────
    print("\nLoading full dataset...")
    df = spark.read.parquet("data/processed/processed_data3")
    df = df.dropna(subset=["clean_text"])

    print("Sample data:")
    df.select("clean_text").show(5, truncate=True)

    # ── 2. TOKENIZE ───────────────────────────────────────────────────────────
    print(" Tokenizing...")
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    df        = tokenizer.transform(df)
    df        = df.filter(size(col("words")) >= 3)

    total = df.count()
    print(f"   Total records: {total:,}")

    print("\nApplying rule-based toxic labeling...")

    df = df \
        .withColumn(
            "has_keyword",
            when(col("clean_text").rlike(KEYWORD_PATTERN), 1).otherwise(0)
        ) \
        .withColumn(
            "has_phrase",
            when(col("clean_text").rlike(TOXIC_PHRASE_PATTERN), 1).otherwise(0)
        ) \
        .withColumn(
            "label",
            when(
                (col("has_keyword") == 1) | (col("has_phrase") == 1),
                1.0
            ).otherwise(0.0)
        )

    print("\nComputing label statistics...")
    label_stats = df.groupBy("label").agg(
    count("*").alias("count"),
    avg("has_keyword").alias("keyword_ratio"),
    avg("has_phrase").alias("phrase_ratio"),
    ).collect()

    label_map      = {int(r["label"]): r.asDict() for r in label_stats}  # ← thêm .asDict()
    toxic_count    = label_map.get(1, {}).get("count", 0)
    nontoxic_count = label_map.get(0, {}).get("count", 0)
    toxic_pct      = toxic_count / total * 100 if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"  LABELING RESULTS")
    print(f"{'='*50}")
    print(f"  Total records : {total:,}")
    print(f"  Toxic (1)     : {toxic_count:,} ({toxic_pct:.1f}%)")
    print(f"  Non-toxic (0) : {nontoxic_count:,} ({100-toxic_pct:.1f}%)")
    print(f"{'='*50}")

    if toxic_pct > 40:
        print("  Toxic ratio cao — threshold quá rộng")
    elif toxic_pct < 3:
        print("  Toxic ratio thấp — có thể mở rộng keyword list")
    else:
        print("  Toxic ratio hợp lý")


    print("\n Building TF-IDF pipeline (numFeatures=500, minDocFreq=10)...")
    hashingTF  = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=500)
    idf        = IDF(inputCol="rawFeatures", outputCol="tfidf", minDocFreq=10)
    normalizer = Normalizer(inputCol="tfidf", outputCol="features", p=2.0)

    pipeline    = Pipeline(stages=[hashingTF, idf, normalizer])
    tfidf_model = pipeline.fit(df)
    df_features = tfidf_model.transform(df)
    df_features.select("clean_text", "label", "features") \
        .write \
        .mode("overwrite") \
        .partitionBy("label") \
        .parquet("data/processed/features3")

    print("   ✓ Saved with partitionBy(label)")

    # ── 7. SAVE TFIDF MODEL ───────────────────────────────────────────────────
    print("\n Saving TF-IDF pipeline model...")
    os.makedirs("outputs/pipeline_model3", exist_ok=True)
    tfidf_model.write().overwrite().save("outputs/pipeline_model3")

    # ── 8. METRICS ───────────────────────────────────────────────────────────
    t_end    = time.time()
    duration = t_end - t_start

    print(f"\n{'='*50}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"  Total records   : {total:,}")
    print(f"  Toxic           : {toxic_count:,} ({toxic_pct:.1f}%)")
    print(f"  Non-toxic       : {nontoxic_count:,} ({100-toxic_pct:.1f}%)")
    print(f"  Pipeline time   : {duration/60:.1f} minutes")
    print(f"  Features dim    : 500 (L2-normalized TF-IDF)")
    print(f"  Labeling method : Rule-based hybrid (keyword + phrase)")
    print(f"  KMeans removed  : Yes (silhouette was -0.0045)")
    print(f"  Output write    : Direct to partitioned Parquet")
    print(f"{'='*50}")

    with open("outputs/metrics_features1.txt", "w") as f:
        f.write(f"total_records={total}\n")
        f.write(f"toxic_count={toxic_count}\n")
        f.write(f"nontoxic_count={nontoxic_count}\n")
        f.write(f"toxic_pct={toxic_pct:.2f}\n")
        f.write(f"num_features=500\n")
        f.write(f"min_doc_freq=10\n")
        f.write(f"labeling_method=rule_based_hybrid\n")
        f.write(f"kmeans_removed=True\n")
        f.write(f"kmeans_silhouette_was=-0.0045\n")
        f.write(f"pipeline_time_minutes={duration/60:.1f}\n")

    print("\nFeature Engineering v5 DONE!")
    print("   Features → data/processed/features3/ (partitioned by label)")
    print("   Model    → outputs/pipeline_model3")
    print("   Metrics  → outputs/metrics_features1.txt")
    print("\n   Chạy tiếp: python src/models/train.py")
    spark.stop()


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
    print("Loading processed data (PARQUET)...")
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
    print(" Tokenizing...")
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    df = tokenizer.transform(df)

    # lọc câu quá ngắn
    df = df.filter(size(col("words")) >= 3)

    # checkpoint để cắt lineage, giải phóng bộ nhớ
    df = df.checkpoint()

    total = df.count()
    print(f"   Loaded {total:,} records")

    # ── 3. TF-IDF + NORMALIZE ─────────────────────────────────────────────────
    print("Building TF-IDF + L2 Normalize...")
    hashingTF  = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
    idf        = IDF(inputCol="rawFeatures", outputCol="tfidf", minDocFreq=5)
    normalizer = Normalizer(inputCol="tfidf", outputCol="features", p=2.0)

    pipeline = Pipeline(stages=[hashingTF, idf, normalizer])
    tfidf_model = pipeline.fit(df)
    df = tfidf_model.transform(df)

    # checkpoint lần 2 sau TF-IDF
    df = df.checkpoint()

    # ── 4. BISECTING K-MEANS ──────────────────────────────────────────────────
    print("Running BisectingKMeans (k=5)...")
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
    print(" Identifying toxic cluster...")

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

    print("Kết quả label:")
    print(f"   Toxic     : {toxic_count:,} ({toxic_pct:.1f}%)")
    print(f"   Non-toxic : {nontoxic_count:,} ({100 - toxic_pct:.1f}%)")

    # ── 7. SAVE FEATURES ──────────────────────────────────────────────────────
    print("Saving features...")
    df_labeled.select("clean_text", "label", "features") \
        .write.mode("overwrite").parquet("data/processed/features")

    # ── 8. SAVE PIPELINE + KMEANS MODEL (để người khác dùng) ─────────────────
    print("Saving pipeline + kmeans models...")
    os.makedirs("outputs/pipeline_model", exist_ok=True)
    os.makedirs("outputs/kmeans_model", exist_ok=True)
    tfidf_model.write().overwrite().save("outputs/pipeline_model")
    km_model.write().overwrite().save("outputs/kmeans_model")
    with open("outputs/toxic_cluster_id.txt", "w") as f:
        f.write(str(toxic_cluster_id))

    print(" Feature engineering DONE!")
    spark.stop()

if __name__ == "__main__":
    run_feature_engineering()