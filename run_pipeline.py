import argparse

import pyspark

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import PipelineModel

transactions_path = "/hdfs/fraud_detection/raw/test_transaction.csv"
identity_path     = "/hdfs/fraud_detection/raw/test_identity.csv"

output_path   = "/hdfs/fraud_detection/data/test_features.parquet"

pipeline_path = "/hdfs/fraud_detection/models/pipeline.model"

num_cols = [
  'TransactionAmt'
]

sql = SparkSession.builder \
    .master("local") \
    .appName("run_pipeline") \
    .getOrCreate()

transactions = sql.read \
 .format("csv") \
 .option("sep", ",") \
 .option("inferSchema", "true") \
 .option("header", "true") \
 .load(transactions_path)

identity = sql.read \
 .format("csv") \
 .option("sep", ",") \
 .option("inferSchema", "true") \
 .option("header", "true") \
 .load(identity_path)

df = transactions.join(identity, on="TransactionID", how="left")

df = df.fillna(-999)
df = df.fillna("na")

for col in num_cols:
  df = df.withColumn(col, df[col] + 1)
  df = df.withColumn(col, F.log(col))

pipeline = PipelineModel.load(pipeline_path)

df = pipeline.transform(df)
df = df.select(["TransactionID", "features"])

df.write \
  .mode("overwrite") \
  .format("parquet") \
  .save(output_path)


