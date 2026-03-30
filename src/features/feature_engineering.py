from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, array_contains, split, lit
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.ml import Pipeline

TOXIC_KEYWORDS = [
    "idiot", "stupid", "hate", "kill", "die", "racist", "moron",
    "dumb", "loser", "trash", "scum", "fool", "jerk", "bastard",
    "terrible", "awful", "disgusting", "pathetic", "worthless"
]

def run_feature_engineering():
    print("🚀 Starting Spark...")
    spark = SparkSession.builder \
        .appName("Feature Engineering") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print("📂 Loading processed data...")
    df = spark.read.csv(
        "data/processed/processed_data",
        header=True
    ).dropna()

    print("🏷️ Auto-labeling toxic comments...")
    # Gán nhãn dựa theo từ khóa toxic
    words_col = split(col("clean_text"), " ")
    
    toxic_condition = lit(False)
    for keyword in TOXIC_KEYWORDS:
        toxic_condition = toxic_condition | array_contains(words_col, keyword)

    df = df.withColumn("label", when(toxic_condition, 1.0).otherwise(0.0))

    # Thống kê nhãn
    total = df.count()
    toxic_count = df.filter(col("label") == 1.0).count()
    print(f"📊 Total: {total} | Toxic: {toxic_count} | Non-toxic: {total - toxic_count}")

    print("🔢 Extracting TF-IDF features...")
    words = split(col("clean_text"), " ")
    df = df.withColumn("words", words)

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    pipeline = Pipeline(stages=[hashingTF, idf])
    model = pipeline.fit(df)
    df_features = model.transform(df)

    print("💾 Saving features...")
    df_features.select("clean_text", "label", "features") \
        .write \
        .mode("overwrite") \
        .parquet("data/processed/features")

    print("✅ Feature engineering DONE!")
    spark.stop()

if __name__ == "__main__":
    run_feature_engineering()