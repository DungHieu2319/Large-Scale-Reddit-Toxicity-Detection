import os
import sys

# ── FIX CHO WINDOWS ─────────────────────────────────────────
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.makedirs("C:/Temp", exist_ok=True)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, size
from pyspark.ml.feature import Tokenizer, StopWordsRemover


def run_preprocessing():
    print("🚀 Starting Spark...")

    spark = SparkSession.builder \
        .appName("Toxic Detection Preprocessing") \
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

    # ── 1. LOAD DATA ────────────────────────────────────────
    print("📂 Loading data (CSV)...")
    df = spark.read.csv("data/raw/reddit_sample.csv", header=True, inferSchema=True)
    df = df.select("body").dropna()

    # ── 2. CLEAN TEXT ───────────────────────────────────────
    print("🧹 Cleaning text...")
    df = df.withColumn("clean_text", lower(col("body")))
    df = df.withColumn("clean_text", regexp_replace(col("clean_text"), r"http\S+", ""))
    df = df.withColumn("clean_text", regexp_replace(col("clean_text"), r"[^a-z\s]", ""))

    # ── 3. TOKENIZE ─────────────────────────────────────────
    print("🔤 Tokenizing...")
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    df = tokenizer.transform(df)

    # ── 4. REMOVE STOPWORDS ─────────────────────────────────
    print("🚫 Removing stopwords...")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    df = remover.transform(df)

    # ── 5. FILTER EMPTY ROWS ────────────────────────────────
    print("⚠️ Filtering empty rows...")
    df = df.filter(size(col("filtered")) > 0)

    # ── 6. SAVE ─────────────────────────────────────────────
    # ✅ bỏ hết count() để tránh crash — chỉ save thẳng
    print("💾 Saving processed data (PARQUET)...")
    df.select("clean_text") \
        .write \
        .mode("overwrite") \
        .parquet("data/processed/processed_data")

    print("✅ PREPROCESS DONE!")
    spark.stop()


if __name__ == "__main__":
    run_preprocessing()
