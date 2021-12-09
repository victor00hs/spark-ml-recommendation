from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.ml.recommendation import ALS 

import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.appName("main").master("local[*]").config("spark.driver.memory", "15g").getOrCreate()

anime_df, ratings_df, valoraciones_df, movie_df, tv_df, union_ratings_valoraciones_df = None, None, None, None, None, None

def load_data():
    # Cargamamos los distintos dataframes
    print('\nCargando dataframes...\n')
    global anime_df, ratings_df, valoraciones_df, movie_df, tv_df, union_ratings_valoraciones_df
    # Crear headers para unir los dataframes ratings con valoraciones_EP
    try:    
        union_headers = StructType([StructField('user_id', IntegerType(), True), StructField('anime_id', IntegerType(), True), StructField('rating', FloatType(), True)])
        anime_df = spark.read.option('encoding', 'UTF-8').csv('data/anime.csv', inferSchema=True, header=True, sep=',')
        # Ponemos en cache este dataframe ya que es muy pesado y asi agilizamos las operaciones realizadas en él
        ratings_df = spark.read.option('encoding', 'UTF-8').csv('data/rating_complete.csv', inferSchema=True, header=True, sep=',')
        valoraciones_df = spark.read.option('encoding', 'UTF-8').csv('data/valoraciones_EP.csv', inferSchema=True, schema=union_headers, header=True, sep=',')
        movie_df, tv_df = anime_df.filter(anime_df.Type == 'Movie'), anime_df.filter(anime_df.Type == 'TV')

        # Pide incorporar las valoraciones de EP al fichero de rating_complete
        union_ratings_valoraciones_df = ratings_df.union(valoraciones_df) 
        print('\nSe han cargado los dataframes...\n')
    except:
        print('Ha ocurrido un error y no se han cargado los ficheros correctamente :\'-(')


def als_recommendation():
    print('\nEntrenando modelo de recomendacion...\n')
    (training, test) = union_ratings_valoraciones_df.randomSplit([0.8, 0.2])
    # Entrenamos el modelo. La estrategia cold start con 'drop' descarata valores NaN en evaluación
    als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(training)
    # Evaluamos el modelo con RMSE
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions) 
    print("Root-mean-square error = " + str(rmse))
    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForUserSubset(valoraciones_df.filter(valoraciones_df.user_id == 666666), 10).show()

if __name__ == '__main__':
    load_data()
    als_recommendation()
