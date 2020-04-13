import pyspark

from pyspark.sql import SparkSession
#from pyspark.sql.types import *
from pyspark.sql import functions as F

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.classification import LogisticRegression

target_column = 'isFraud'

cat_cols = [
    'ProductCD',
    'card4',
    #'P_emaildomain',
    #'R_emaildomain'
]

num_cols = [
    'TransactionAmt'
]

v_cols = []
for i in range(1, 339):
    v_cols.append('V{}'.format(i))

c_cols = []
for i in range(1, 14):
    c_cols.append('C{}'.format(i))

selected_cols  = []
selected_cols += num_cols
selected_cols += cat_cols
#selected_cols += c_cols
#selected_cols += v_cols
selected_cols += [target_column]

sql = SparkSession.builder \
    .master("local") \
    .appName("transform_data") \
    .getOrCreate()

#df = sql.read \
#    .format("parquet") \
#    .load("/hdfs/fraud_detection/data/train_transaction.parquet")

df = sql.read \
    .format("csv") \
    .option("sep", ",") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .load("/hdfs/fraud_detection/raw/train_transaction.csv")

df = df.select(selected_cols)

for col in num_cols:
    df = df.withColumn(col, df[col] + 1)
    df = df.withColumn(col, F.log(col))

vec_num = VectorAssembler(inputCols=num_cols, outputCol="num_cols")

df = vec_num.transform(df)

indexers = []

cat_cols_idx = map(lambda x: "{}_idx".format(x), cat_cols)
cat_cols_idx = list(cat_cols_idx)

for col, col_idx in zip(cat_cols, cat_cols_idx):
    indexer = StringIndexer(inputCol=col, outputCol=col_idx, handleInvalid="skip")
    indexer = indexer.fit(df)
    
    df = indexer.transform(df)

    indexers += [indexer]

cat_cols_enc = map(lambda x: "{}_enc".format(x), cat_cols)
cat_cols_enc = list(cat_cols_enc)

encoder = OneHotEncoderEstimator(inputCols=cat_cols_idx, outputCols=cat_cols_enc)
encoder = encoder.fit(df)

df = encoder.transform(df)

input_cols  = ["num_cols"]
input_cols += cat_cols_enc

assembler = VectorAssembler(inputCols=input_cols, outputCol="features")

df = assembler.transform(df)

df = df.select(["features", "isFraud"])

print(df.columns)
print(df.show(5))

df.write \
  .mode("overwrite") \
  .format("parquet") \
  .save("/hdfs/fraud_detection/data/train_features.parquet")

"""
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
"""