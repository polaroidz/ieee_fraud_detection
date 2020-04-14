import pyspark

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.classification import LogisticRegression

# Arguments

dataset_path   = "/hdfs/fraud_detection/data/train_features.parquet"
model_out_path = "/hdfs/fraud_detection/models/lm.model"

# Load Data

sql = SparkSession.builder \
    .master("local") \
    .appName("feature_engineering") \
    .getOrCreate()

df = sql.read \
  .format("parquet") \
  .load(dataset_path)

# Oversampling

positive = df.where(df.isFraud == 1)
negative = df.where(df.isFraud == 0)

fraction = negative.count() / positive.count()

positive = positive.sample(withReplacement=True, fraction=fraction, seed=42)

df = positive.union(negative)

# Model

lr = LogisticRegression(
  featuresCol="features", 
  labelCol="isFraud",
  predictionCol="predictions",
  maxIter=10,
  regParam=0.0,
  elasticNetParam=0.0,
  threshold=0.5
)
lr = lr.fit(df)

print("MODEL SUMMARY")

df = lr.transform(df)

summary = lr.summary

print("Labels")
print(summary.labels)

print("Accuracy")
print(summary.accuracy)

print("Precision by Label")
print(summary.precisionByLabel)

print("Recall by Label")
print(summary.recallByLabel)

print("False Positve Rate")
print(summary.falsePositiveRateByLabel)

print("True Positive Rate by Label")
print(summary.truePositiveRateByLabel)

print("Area Under ROC")
print(summary.areaUnderROC)

roc = summary.roc.toPandas()
roc.to_csv("./roc.csv", index=False)

lr.write() \
  .overwrite() \
  .save(model_out_path)

