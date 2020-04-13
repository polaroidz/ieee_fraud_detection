import pyspark

from pyspark.sql import SparkSession

sql = SparkSession.builder \
    .master("local") \
    .appName("fn_ingest_raw_data") \
    .getOrCreate()

df = sql.read \
    .format("csv") \
    .option("sep", ",") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .load("/hdfs/fraud_detection/raw/train_transaction.csv")

df.write \
    .mode("overwrite") \
    .format("parquet") \
    .save("/hdfs/fraud_detection/data/train_transaction.parquet")
