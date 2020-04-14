import argparse

import pyspark

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import PCA

from pyspark.ml.classification import LogisticRegression

# Arguments

dataset_path   = "/hdfs/fraud_detection/raw/train_transaction.csv"
output_path  = "/hdfs/fraud_detection/data/train_features.parquet"
model_out_path = "/hdfs/fraud_detection/models/pipeline.model"

# Hyperparameters

V_PCA_K = 35

# Utilities

target_column = 'isFraud'

cat_cols = [
  'ProductCD',
  'card1',
  'card2',
  'card3',
  'card4',
  'card5',
  'card6',
  'P_emaildomain',
  'R_emaildomain'
]

num_cols = [
  'TransactionAmt'
]

v_cols = []
for i in range(12, 339):
  v_cols.append('V{}'.format(i))

c_cols = []
for i in range(1, 14):
  c_cols.append('C{}'.format(i))

# Loading Data

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

# Numerical Columns

for col in num_cols:
  df = df.withColumn(col, df[col] + 1)
  df = df.withColumn(col, F.log(col))

vec_num = VectorAssembler(inputCols=num_cols, outputCol="num_cols")

# Categorical Columns

cat_cols_idx = map(lambda x: "{}_idx".format(x), cat_cols)
cat_cols_idx = list(cat_cols_idx)

cat_cols_enc = map(lambda x: "{}_enc".format(x), cat_cols)
cat_cols_enc = list(cat_cols_enc)

cat_cols_nan = map(lambda x: "{}_nan".format(x), cat_cols)
cat_cols_nan = list(cat_cols_nan)

cat_indexers = []

for col, col_idx in zip(cat_cols, cat_cols_idx):
  indexer = StringIndexer(inputCol=col, outputCol=col_idx, handleInvalid="keep")
  cat_indexers += [indexer]

cat_imputer = Imputer(inputCols=cat_cols_idx, outputCols=cat_cols_nan, strategy="median")
cat_encoder = OneHotEncoderEstimator(inputCols=cat_cols_nan, outputCols=cat_cols_enc, handleInvalid="keep")

# V Columns

v_cols_nan = map(lambda x: "{}_nan".format(x), v_cols)
v_cols_nan = list(v_cols_nan)

v_imputer = Imputer(inputCols=v_cols, outputCols=v_cols_nan, strategy="median")
v_assembler = VectorAssembler(inputCols=v_cols_nan, outputCol="v_cols", handleInvalid="skip")
v_minmax = MinMaxScaler(inputCol="v_cols", outputCol="v_scaled")
v_pca = PCA(k=V_PCA_K, inputCol="v_scaled", outputCol="v_pca")

# Assembling Columns

input_cols  = ["num_cols"]
input_cols  = ["v_pca"]
input_cols += cat_cols_enc

assembler = VectorAssembler(inputCols=input_cols, outputCol="features")

# Pipeline

pipeline = Pipeline(stages=[
  vec_num,
  *cat_indexers,
  cat_imputer,
  cat_encoder,
  v_imputer,
  v_assembler,
  v_minmax,
  v_pca,
  assembler
])

pipeline = pipeline.fit(df)

pipeline.write() \
  .overwrite() \
  .save(model_out_path)

df = pipeline.transform(df)
df = df.select(["features", "isFraud"])

df.write \
  .mode("overwrite") \
  .format("parquet") \
  .save(output_path)
