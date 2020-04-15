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

transactions_path = "/hdfs/fraud_detection/raw/train_transaction.csv"
identity_path     = "/hdfs/fraud_detection/raw/train_identity.csv"

output_path  = "/hdfs/fraud_detection/data/train_features.parquet"
model_out_path = "/hdfs/fraud_detection/models/pipeline.model"

# Hyperparameters

V_PCA_K  = 35
ID_PCA_K = 12 

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
  'R_emaildomain',
  'DeviceType',
  'DeviceInfo'
]

num_cols = [
  'TransactionAmt'
]

v_cols = []
for i in range(12, 339):
  v_cols.append('V{}'.format(i))

id_cols = []
for i in range(1, 38):
  if i < 10:
    id_cols.append('id_0{}'.format(i))
  else:
    id_cols.append('id_{}'.format(i))

c_cols = []
for i in range(1, 14):
  c_cols.append('C{}'.format(i))


# Loading Data

sql = SparkSession.builder \
  .master("local") \
  .appName("train_pipeline") \
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

# ID Columns

id_cols_nan = map(lambda x: "{}_nan".format(x), id_cols)
id_cols_nan = list(id_cols_nan)

id_imputer = Imputer(inputCols=id_cols, outputCols=id_cols_nan, strategy="median")
id_assembler = VectorAssembler(inputCols=id_cols_nan, outputCol="id_cols", handleInvalid="skip")
id_minmax = MinMaxScaler(inputCol="id_cols", outputCol="id_scaled")
id_pca = PCA(k=ID_PCA_K, inputCol="id_scaled", outputCol="id_pca")

# C Columns

c_cols_nan = map(lambda x: "{}_nan".format(x), c_cols)
c_cols_nan = list(c_cols_nan)

c_imputer = Imputer(inputCols=c_cols, outputCols=c_cols_nan, strategy="median")
c_assembler = VectorAssembler(inputCols=c_cols_nan, outputCol="c_cols", handleInvalid="skip")
c_minmax = MinMaxScaler(inputCol="c_cols", outputCol="c_scaled")

# Assembling Columns

input_cols  = ["num_cols"]
input_cols += ["v_pca"]
#input_cols += ["id_pca"]
input_cols += ["c_scaled"]
input_cols += cat_cols_enc

assembler = VectorAssembler(inputCols=input_cols, outputCol="features")

# Pipeline

pipeline = Pipeline(stages=[
  vec_num,
  *cat_indexers,
  cat_imputer,
  cat_encoder,
  c_imputer,
  c_assembler,
  c_minmax,
  v_imputer,
  v_assembler,
  v_minmax,
  v_pca,
#  id_imputer,
#  id_assembler,
#  id_minmax,
#  id_pca,
  assembler
])

pipeline = pipeline.fit(df)

pipeline.write() \
  .overwrite() \
  .save(model_out_path)

df = pipeline.transform(df)
df = df.select(["TransactionID", "features", "isFraud"])

df.write \
  .mode("overwrite") \
  .format("parquet") \
  .save(output_path)
