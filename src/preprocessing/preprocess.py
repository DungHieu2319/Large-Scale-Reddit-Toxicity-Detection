from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover


def run_preprocessing():

    print("🚀 Starting Spark...")

    spark = SparkSession.builder \
        .appName("Toxic Detection") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/javax.security.auth=ALL-UNNAMED") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print("📂 Loading data...")

    df = spark.read.csv(
        "data/raw/reddit_sample.csv",
        header=True,
        multiLine=True,
        escape='"'
    )

    df = df.select("body").dropna().limit(200000)

    print("🧹 Cleaning text (FAST)...")

    df = df.withColumn("clean_text", lower(col("body")))
    df = df.withColumn("clean_text", regexp_replace(col("clean_text"), r"http\S+", ""))
    df = df.withColumn("clean_text", regexp_replace(col("clean_text"), r"[^a-z\s]", ""))

    print("🔤 Tokenizing...")

    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    df = tokenizer.transform(df)

    print("🧠 Removing stopwords...")

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    df = remover.transform(df)

    print("💾 Saving...")

    df.select("clean_text") \
        .write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("data/processed/processed_data")

    print("✅ DONE!")


if __name__ == "__main__":
    run_preprocessing()