import pyspark

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.classification import LogisticRegressionModel

dataset_path   = "/hdfs/fraud_detection/data/test_features.parquet"
model_path = "/hdfs/fraud_detection/models/lm.model"

sql = SparkSession.builder \
    .master("local") \
    .appName("run_model") \
    .getOrCreate()

df = sql.read \
  .format("parquet") \
  .load(dataset_path)

lm = LogisticRegressionModel.load(model_path)

df = lm.transform(df)

def extract_probs(row):
  return (row.TransactionID,) + tuple(row.probability.toArray().tolist())

df = df.rdd.map(extract_probs).toDF(["TransactionID", "notIsFraud", "isFraud"])
df = df.select(["TransactionID", "isFraud"])

submission = df.toPandas()
submission.to_csv("./submission.csv", index=False)

