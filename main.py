from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("main").master("local[*]").getOrCreate()
anime_df = spark.read.csv('datos/cards.csv', inferSchema=True, header=True, sep='|')