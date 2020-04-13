import pyspark

from pyspark.sql import SparkSession

from pyspark.ml.classification import LogisticRegression

sql = SparkSession.builder \
    .master("local") \
    .appName("feature_engineering") \
    .getOrCreate()

df = sql.read \
  .format("parquet") \
  .load("/hdfs/fraud_detection/data/train_features.parquet")

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

df = lr.transform(df)

summary = lr.summary

#print(df.columns)
print(df.show(5))

roc = summary.roc
print("ROC COUNT:" + str(roc.count()))

print("areaUnderROC: " + str(summary.areaUnderROC))

#print("There are", df.count(), "lines")
#print(df.cols)
