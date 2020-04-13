import pyspark

from pyspark.sql import SparkSession

from pyspark.sql import functions as F

from pyspark.ml.classification import LogisticRegression

# Hyperparameters

NEGATIVE_SAMPLE_FRAC = 0.05

sql = SparkSession.builder \
    .master("local") \
    .appName("feature_engineering") \
    .getOrCreate()

df = sql.read \
  .format("parquet") \
  .load("/hdfs/fraud_detection/data/train_features.parquet")

df = df.sampleBy("isFraud", fractions={0: NEGATIVE_SAMPLE_FRAC, 1: 1}, seed=42)

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

#print(df.columns)
print(df.show(5))

#roc = summary.roc
#print("ROC COUNT:" + str(roc.count()))
