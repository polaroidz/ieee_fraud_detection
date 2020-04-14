import argparse

import pyspark

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import PipelineModel

dataset_path  = "/hdfs/fraud_detection/raw/test_transaction.csv"
output_path   = "/hdfs/fraud_detection/data/test_features.parquet"
pipeline_path = "/hdfs/fraud_detection/models/pipeline.model"

sql = SparkSession.builder \
    .master("local") \
    .appName("feature_engineering") \
    .getOrCreate()

df = sql.read \
    .format("csv") \
    .option("sep", ",") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .load(dataset_path)

pipeline = PipelineModel.load(pipeline_path)

df = pipeline.transform(df)
df = df.select(["features"])

df.write \
  .mode("overwrite") \
  .format("parquet") \
  .save(output_path)


