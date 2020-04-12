import pyspark

from pyspark.sql import SparkSession
#from pyspark.sql.types import *
from pyspark.sql import functions as F

from pyspark.ml.feature import VectorAssembler

target_column = 'isFraud'

categorical_columns = [
    'ProductCD',
    'card4',
    #'P_emaildomain',
    #'R_emaildomain'
]

numerical_columns = [
    'TransactionAmt'
]

v_columns = []
for i in range(1, 339):
    v_columns.append('V{}'.format(i))

c_columns = []
for i in range(1, 14):
    c_columns.append('C{}'.format(i))

selected_columns  = []
selected_columns += numerical_columns
selected_columns += categorical_columns
#selected_columns += c_columns
#selected_columns += v_columns
selected_columns += [target_column]

if __name__ == '__main__':
    sql = SparkSession.builder \
        .master("local") \
        .appName("fn_ingest_raw_data") \
        .getOrCreate()

    df = sql.read \
        .format("parquet") \
        .load("/hdfs/fraud_detection/data/train_transaction.parquet")

    df = df.select(selected_columns)

    for column in numerical_columns:
        df = df.withColumn(column, df[column] + 1)
        df = df.withColumn(column, F.log(column))

    vec_numerical = VectorAssembler(inputCols=numerical_columns, outputCol="numerical_columns")
    df = vec_numerical.transform(df)

    print(df.columns)
    print(df.show(5))

    #print("There are", df.count(), "lines")
    #print(df.columns)
