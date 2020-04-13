import pyspark

from pyspark.sql import SparkSession
#from pyspark.sql.types import *
from pyspark.sql import functions as F

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import PCA

from pyspark.ml.classification import LogisticRegression

# Hyperparameters

V_PCA_K = 35

# Utilities

target_column = 'isFraud'

cat_cols = [
    'ProductCD',
    'card4',
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

selected_cols  = []
selected_cols += num_cols
selected_cols += cat_cols
#selected_cols += c_cols
selected_cols += v_cols
selected_cols += [target_column]

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
    .load("/hdfs/fraud_detection/raw/train_transaction.csv")

df = df.select(selected_cols)

# Numerical Columns

for col in num_cols:
    df = df.withColumn(col, df[col] + 1)
    df = df.withColumn(col, F.log(col))

vec_num = VectorAssembler(inputCols=num_cols, outputCol="num_cols")

df = vec_num.transform(df)

# Categorical Columns

cat_indexers = []

cat_cols_idx = map(lambda x: "{}_idx".format(x), cat_cols)
cat_cols_idx = list(cat_cols_idx)

for col, col_idx in zip(cat_cols, cat_cols_idx):
    indexer = StringIndexer(inputCol=col, outputCol=col_idx, handleInvalid="skip")
    indexer = indexer.fit(df)
    
    df = indexer.transform(df)

    cat_indexers += [indexer]

cat_cols_enc = map(lambda x: "{}_enc".format(x), cat_cols)
cat_cols_enc = list(cat_cols_enc)

encoder = OneHotEncoderEstimator(inputCols=cat_cols_idx, outputCols=cat_cols_enc)
encoder = encoder.fit(df)

df = encoder.transform(df)

# V Columns

v_cols_nan = map(lambda x: "{}_nan".format(x), v_cols)
v_cols_nan = list(v_cols_nan)

v_imputer = Imputer(inputCols=v_cols, outputCols=v_cols_nan, strategy="median")
v_imputer = v_imputer.fit(df)

df = v_imputer.transform(df)

v_assembler = VectorAssembler(inputCols=v_cols_nan, outputCol="v_cols", handleInvalid="skip")

df = v_assembler.transform(df)

v_minmax = MinMaxScaler(inputCol="v_cols", outputCol="v_scaled")
v_minmax = v_minmax.fit(df)

df = v_minmax.transform(df)

v_pca = PCA(k=V_PCA_K, inputCol="v_scaled", outputCol="v_pca")
v_pca = v_pca.fit(df)

df = v_pca.transform(df)

# Assembling Columns

input_cols  = ["num_cols"]
input_cols  = ["v_pca"]
input_cols += cat_cols_enc

assembler = VectorAssembler(inputCols=input_cols, outputCol="features")

df = assembler.transform(df)

df = df.select(["features", "isFraud"])

df.write \
  .mode("overwrite") \
  .format("parquet") \
  .save("/hdfs/fraud_detection/data/train_features.parquet")
