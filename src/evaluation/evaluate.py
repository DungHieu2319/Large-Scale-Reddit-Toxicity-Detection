import sys

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import os
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
def run_evaluation():
    print("🚀 Starting Spark...")
    spark = SparkSession.builder \
        .appName("Toxic Detection - Evaluation") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print("📂 Loading features...")
    df = spark.read.parquet("data/processed/features")

    print("✂️ Splitting data 80/20...")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    print("🤖 Training model...")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
    model = lr.fit(train_df)
    predictions = model.transform(test_df)

    print("\n📊 ===== EVALUATION RESULTS =====")

    auc_eval = BinaryClassificationEvaluator(labelCol="label")
    auc = auc_eval.evaluate(predictions)
    print(f"📈 AUC:       {auc:.4f}")

    acc_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    acc = acc_eval.evaluate(predictions)
    print(f"✅ Accuracy:  {acc:.4f}")

    f1_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
    f1 = f1_eval.evaluate(predictions)
    print(f"🎯 F1-Score:  {f1:.4f}")

    prec_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedPrecision")
    prec = prec_eval.evaluate(predictions)
    print(f"🔍 Precision: {prec:.4f}")

    rec_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedRecall")
    rec = rec_eval.evaluate(predictions)
    print(f"🔁 Recall:    {rec:.4f}")

    print("\n🗂️ Confusion Matrix:")
    preds_and_labels = predictions.select("prediction", "label").rdd \
        .map(lambda r: (float(r[0]), float(r[1])))
    metrics = MulticlassMetrics(preds_and_labels)
    cm = metrics.confusionMatrix().toArray()
    print(f"  TN: {int(cm[0][0])}  FP: {int(cm[0][1])}")
    print(f"  FN: {int(cm[1][0])}  TP: {int(cm[1][1])}")

    print("\n✅ Evaluation DONE!")
    spark.stop()

if __name__ == "__main__":
    run_evaluation()