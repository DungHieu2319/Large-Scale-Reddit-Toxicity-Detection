from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os

def run_training():
    print("🚀 Starting Spark...")
    spark = SparkSession.builder \
        .appName("Toxic Detection - Training") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print("📂 Loading features...")
    df = spark.read.parquet("data/processed/features")

    print("✂️ Splitting data 80/20...")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Train: {train_df.count()} | Test: {test_df.count()}")

    print("🤖 Training Logistic Regression...")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=10
    )
    model = lr.fit(train_df)

    print("🔍 Evaluating...")
    predictions = model.transform(test_df)

    evaluator = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator.evaluate(predictions)
    print(f"📈 AUC: {auc:.4f}")

    print("💾 Saving model...")
    os.makedirs("outputs/model", exist_ok=True)
    model.save("outputs/model/lr_toxic_model")

    print("✅ Training DONE!")
    spark.stop()

if __name__ == "__main__":
    run_training()