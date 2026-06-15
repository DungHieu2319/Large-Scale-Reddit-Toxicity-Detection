import argparse
import json
import os
import sys
import time

import pyarrow.parquet as parquet

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.makedirs("D:/SparkTemp", exist_ok=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(PROJECT_ROOT)

from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.common import _py2java
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


DEFAULT_MODEL_PATH = "outputs/model_tong/lr_toxic_model_tong"
DEFAULT_FEATURE_PATHS = [
    "data/processed/features",
    "data/processed/features1",
    "data/processed/features2",
    "data/processed/features3",
]
DEFAULT_METRICS_PATH = "outputs/evaluation_metrics0.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Spark Logistic Regression model."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--feature-path",
        action="append",
        dest="feature_paths",
        help="Feature directory. May be supplied more than once.",
    )
    parser.add_argument("--metrics-path", default=DEFAULT_METRICS_PATH)
    return parser.parse_args()


def create_spark_session():
    return (
        SparkSession.builder
        .appName("Toxic Detection - Saved Model Evaluation")
        .master("local[4]")
        .config("spark.driver.memory", "6g")
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.local.dir", "D:/SparkTemp")
        .config("spark.sql.files.maxPartitionBytes", "64m")
        .config("spark.sql.shuffle.partitions", "256")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "128m")
        .config(
            "spark.driver.extraJavaOptions",
            "-Djava.io.tmpdir=D:/SparkTemp "
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
            "--add-opens=java.base/java.lang=ALL-UNNAMED",
        )
        .getOrCreate()
    )


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def load_features(spark, feature_paths):
    dataframes = []

    print("Loading evaluation features...")
    for path in feature_paths:
        if not os.path.exists(path):
            print(f"  Skip missing path: {path}")
            continue

        dataframe = (
            spark.read.parquet(path)
            .select("features", "label")
            .dropna(subset=["features", "label"])
        )
        dataframes.append(dataframe)
        print(f"  Loaded: {path}")

    if not dataframes:
        raise FileNotFoundError("No evaluation feature directory was found.")

    combined = dataframes[0]
    for dataframe in dataframes[1:]:
        combined = combined.unionByName(dataframe)

    return combined


def load_logistic_regression_model(spark, model_path):
    """Load a saved LR model, bypassing broken Hadoop NativeIO on Windows."""
    if os.name != "nt":
        return LogisticRegressionModel.load(model_path)

    print("  Using portable Windows model loader (no Hadoop NativeIO).")

    metadata_path = os.path.join(model_path, "metadata", "part-00000")
    data_files = [
        os.path.join(model_path, "data", filename)
        for filename in os.listdir(os.path.join(model_path, "data"))
        if filename.startswith("part-")
    ]
    if not data_files:
        raise FileNotFoundError(f"Model coefficient data not found: {model_path}")

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    model_data = parquet.read_table(data_files[0]).to_pylist()[0]
    coefficients = Vectors.dense(model_data["coefficientMatrix"]["values"])
    intercept = float(model_data["interceptVector"]["values"][0])
    java_coefficients = _py2java(spark.sparkContext, coefficients)

    java_model = spark._jvm.org.apache.spark.ml.classification.LogisticRegressionModel(
        metadata["uid"],
        java_coefficients,
        intercept,
    )

    model = LogisticRegressionModel(java_model)
    valid_params = {
        name: value
        for name, value in metadata.get("paramMap", {}).items()
        if model.hasParam(name)
    }
    model._set(**valid_params)
    model._transfer_params_to_java()
    return model


def run_evaluation():
    args = parse_args()
    feature_paths = args.feature_paths or DEFAULT_FEATURE_PATHS

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Saved model not found: {args.model_path}")

    started_at = time.time()
    print("Starting Spark evaluation...")
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        print(f"Loading saved model: {args.model_path}")
        model = load_logistic_regression_model(spark, args.model_path)

        dataframe = load_features(spark, feature_paths)

        print("Running predictions with the saved model...")
        predictions = model.transform(dataframe).select(
            col("label").cast("double"),
            col("prediction").cast("double"),
            "rawPrediction",
        )

        counts = {
            (int(row["label"]), int(row["prediction"])): row["count"]
            for row in predictions.groupBy("label", "prediction").count().collect()
        }

        tn = counts.get((0, 0), 0)
        fp = counts.get((0, 1), 0)
        fn = counts.get((1, 0), 0)
        tp = counts.get((1, 1), 0)
        total = tn + fp + fn + tp

        accuracy = safe_divide(tp + tn, total)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        specificity = safe_divide(tn, tn + fp)

        auc_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        )
        auc = auc_evaluator.evaluate(predictions)
        duration = time.time() - started_at

        print("\n===== SAVED MODEL EVALUATION =====")
        print(f"Model       : {args.model_path}")
        print(f"Records     : {total:,}")
        print(f"AUC         : {auc:.4f}")
        print(f"Accuracy    : {accuracy:.4f}")
        print(f"Precision   : {precision:.4f}")
        print(f"Recall      : {recall:.4f}")
        print(f"F1-score    : {f1:.4f}")
        print(f"Specificity : {specificity:.4f}")
        print("\nConfusion matrix:")
        print(f"TN={tn:,}  FP={fp:,}")
        print(f"FN={fn:,}  TP={tp:,}")
        print(f"Duration    : {duration / 60:.1f} minutes")
        print(
            "WARNING: This model was trained on these same feature paths, "
            "so these are training-set metrics, not independent test metrics."
        )

        metrics_dir = os.path.dirname(args.metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)

        with open(args.metrics_path, "w", encoding="utf-8") as metrics_file:
            metrics_file.write(f"model_path={args.model_path}\n")
            metrics_file.write(f"feature_paths={','.join(feature_paths)}\n")
            metrics_file.write("evaluation_scope=training_data\n")
            metrics_file.write(f"records={total}\n")
            metrics_file.write(f"auc={auc:.6f}\n")
            metrics_file.write(f"accuracy={accuracy:.6f}\n")
            metrics_file.write(f"precision={precision:.6f}\n")
            metrics_file.write(f"recall={recall:.6f}\n")
            metrics_file.write(f"f1={f1:.6f}\n")
            metrics_file.write(f"specificity={specificity:.6f}\n")
            metrics_file.write(f"tn={tn}\n")
            metrics_file.write(f"fp={fp}\n")
            metrics_file.write(f"fn={fn}\n")
            metrics_file.write(f"tp={tp}\n")
            metrics_file.write(f"duration_minutes={duration / 60:.2f}\n")

        print(f"Metrics saved to: {args.metrics_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    run_evaluation()