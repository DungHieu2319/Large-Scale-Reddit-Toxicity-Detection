"""
train_on_bert_embeddings.py
===========================
Train Spark MLlib classifiers (LR, RF, GBT) trên BERT embeddings 768-dim.
Chạy SAU khi extract_embeddings.py đã xong.

So sánh 3 approach:
  1. TF-IDF 500-dim + Spark MLlib   (feature_engineering_v5 → train_v3)
  2. BERT embeddings 768-dim + Spark MLlib  (file này)
  3. Fine-tuned DistilBERT/BERT     (finetune_bert.py)
"""

import os
import sys
import time

os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(PROJECT_ROOT)

from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler
from pyspark.storagelevel import StorageLevel
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)


def create_spark_session():
    return (
        SparkSession.builder
        .appName("Train on BERT Embeddings")
        .master("local[6]")
        .config("spark.driver.memory",        "12g")
        .config("spark.driver.maxResultSize",  "3g")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size",    "3g")
        .config("spark.memory.storageFraction", "0.3")
        .config("spark.local.dir", "D:/SparkTemp")
        .config("spark.sql.adaptive.enabled",                      "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled",   "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.driver.extraJavaOptions",
                "-Djava.io.tmpdir=D:/SparkTemp "
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                "--add-opens=java.base/java.lang=ALL-UNNAMED")
        .getOrCreate()
    )


def evaluate_model(predictions, test_count, model_name, train_time):
    auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    ).evaluate(predictions)

    acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="accuracy",
    ).evaluate(predictions)

    f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="f1",
    ).evaluate(predictions)

    precision = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="precisionByLabel",
    ).evaluate(predictions)

    recall = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="recallByLabel",
    ).evaluate(predictions)

    cm     = predictions.groupBy("label", "prediction").count().collect()
    cm_map = {(int(r["label"]), int(r["prediction"])): r["count"] for r in cm}
    TP = cm_map.get((1, 1), 0)
    FP = cm_map.get((0, 1), 0)
    TN = cm_map.get((0, 0), 0)
    FN = cm_map.get((1, 0), 0)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Train time : {train_time:.1f}s")
    print(f"  AUC-ROC    : {auc:.4f}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  Confusion Matrix (test: {test_count:,}):")
    print(f"              Pred 0    Pred 1")
    print(f"  Actual 0  : {TN:>8,}  {FP:>8,}  ← Non-toxic")
    print(f"  Actual 1  : {FN:>8,}  {TP:>8,}  ← Toxic")
    print(f"{'='*55}")

    return {
        "name": model_name, "auc": auc, "acc": acc, "f1": f1,
        "precision": precision, "recall": recall,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "train_time": train_time,
    }


def run_training():
    t_start = time.time()
    print("Training on BERT-BASE Embeddings (768-dim)")
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    # ── 1. LOAD BERT EMBEDDINGS ───────────────────────────────────────────────
    print("\nLoading BERT-BASE embeddings...")

    # Phải khớp DATA_PERCENT trong extract_embeddings_on_bert.py.
    DATA_PERCENT = 10
    BERT_PATH = f"data/processed/bert_base_embeddings_{DATA_PERCENT}pct"
    out_folder = "outputs/model_bert_base"
    out_metrics = "outputs/metrics_bert_base_spark.txt"
    if not os.path.exists(BERT_PATH):
        print(f"{BERT_PATH} không tồn tại!")
        print("   Chạy extract_embeddings.py trước")
        spark.stop()
        return

    df_raw = spark.read.parquet(BERT_PATH)

    # Convert 768 cột số → Spark DenseVector
    emb_cols = [f"emb_{i}" for i in range(768)]

    # Dùng VectorAssembler chạy trực tiếp trên JVM, tránh quá tải RAM do UDF Python chuyển dữ liệu sang Python worker
    assembler = VectorAssembler(inputCols=emb_cols, outputCol="features")
    df = assembler.transform(df_raw).select("label", "features").dropna()

    # ── 2. CLASS STATS + WEIGHT ───────────────────────────────────────────────
    stats    = df.groupBy("label").count().collect()
    stat_map = {int(r["label"]): r["count"] for r in stats}
    toxic    = stat_map.get(1, 0)
    nontoxic = stat_map.get(0, 0)
    total    = toxic + nontoxic
    weight   = round(nontoxic / toxic, 2) if toxic > 0 else 1.0

    print(f"\nData loaded:")
    print(f"   Total    : {total:,}")
    print(f"   Toxic    : {toxic:,} ({toxic/total*100:.1f}%)")
    print(f"   Non-toxic: {nontoxic:,} ({nontoxic/total*100:.1f}%)")
    print(f"   Features : 768-dim BERT embeddings")
    print(f"   Weight   : Toxic×{weight}")

    df = df.withColumn(
        "classWeight",
        when(col("label") == 1.0, weight).otherwise(1.0)
    )

    # ── 3. TRAIN / TEST SPLIT ─────────────────────────────────────────────────
    print("\nSplitting 80/20...")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Dùng persist(MEMORY_AND_DISK) thay cho checkpoint để tránh lưu nhiều file xuống temp
    train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
    test_df  = test_df.persist(StorageLevel.MEMORY_AND_DISK)

    train_n = train_df.count()
    test_n  = test_df.count()
    print(f"   Train: {train_n:,} | Test: {test_n:,}")

    results = []

    # ── 4. LOGISTIC REGRESSION ────────────────────────────────────────────────
    # LR trên BERT embeddings: tốt hơn TF-IDF vì embeddings đã capture ngữ nghĩa
    print("\n[1/3] Logistic Regression on BERT embeddings...")
    t0 = time.time()
    lr = LogisticRegression(
        featuresCol="features", labelCol="label", weightCol="classWeight",
        maxIter=20, regParam=0.01, elasticNetParam=0.1, tol=1e-4,
    )
    lr_model  = lr.fit(train_df)
    lr_preds  = lr_model.transform(test_df)
    lr_result = evaluate_model(lr_preds, test_n, "LR + BERT embeddings", time.time()-t0)
    results.append(lr_result)

    # ── 5. RANDOM FOREST ──────────────────────────────────────────────────────
    print("\n🌲 [2/3] Random Forest on BERT embeddings...")
    t0 = time.time()
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="label",
        numTrees=50, maxDepth=10,
        featureSubsetStrategy="sqrt",
        subsamplingRate=0.8, seed=42,
    )
    rf_model  = rf.fit(train_df)
    rf_preds  = rf_model.transform(test_df)
    rf_result = evaluate_model(rf_preds, test_n, "RF + BERT embeddings", time.time()-t0)
    results.append(rf_result)

    # ── 6. GBT ────────────────────────────────────────────────────────────────
    print("\n[3/3] GBT on BERT embeddings...")
    t0 = time.time()
    gbt = GBTClassifier(
        featuresCol="features", labelCol="label",
        maxIter=20, maxDepth=5, stepSize=0.1,
        subsamplingRate=0.8, seed=42,
    )
    gbt_model  = gbt.fit(train_df)
    gbt_preds  = gbt_model.transform(test_df)
    gbt_result = evaluate_model(gbt_preds, test_n, "GBT + BERT embeddings", time.time()-t0)
    results.append(gbt_result)

    # ── 7. SUMMARY ────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  BERT EMBEDDINGS + SPARK MLlib — COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Model':<28} {'AUC':>6} {'F1':>6} {'Acc':>6} {'Time':>8}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['name']:<28} {r['auc']:>6.4f} {r['f1']:>6.4f} "
              f"{r['acc']:>6.4f} {r['train_time']:>6.1f}s")
    print(f"{'='*60}")

    best = max(results, key=lambda r: r["auc"])
    print(f"\n  🏆 Best: {best['name']} (AUC={best['auc']:.4f})")
    print(f"  ⏱️  Total: {total_time/60:.1f} minutes")

    # ── 8. SAVE ───────────────────────────────────────────────────────────────
    os.makedirs(out_folder, exist_ok=True)
    lr_model.write().overwrite().save(f"{out_folder}/lr_model")
    rf_model.write().overwrite().save(f"{out_folder}/rf_model")
    gbt_model.write().overwrite().save(f"{out_folder}/gbt_model")

    with open(out_metrics, "w") as f:
        f.write(f"feature_type=bert_base_embeddings_768dim\n")
        f.write(f"total={total}\ntrain={train_n}\ntest={test_n}\n")
        f.write(f"total_time_minutes={total_time/60:.1f}\n\n")
        for r in results:
            prefix = r["name"].lower().replace(" ", "_").replace("+","").replace("__","_")
            for k, v in r.items():
                if k != "name":
                    f.write(f"{prefix}_{k}={v}\n")

    print(f"\nTraining on BERT-BASE embeddings DONE!")
    print(f"   Models  → {out_folder}/")
    print(f"   Metrics → {out_metrics}")
    spark.stop()


if __name__ == "__main__":
    run_training()
