from typing import Text
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.ml.recommendation import ALS
import numpy as np

import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.appName("main").master("local[*]").config("spark.driver.memory", "15g").getOrCreate()

anime_df, ratings_df, valoraciones_df, movie_df, tv_df, union_ratings_valoraciones_df = None, None, None, None, None, None
ratings_movies_df = None

def load_data():
    # Cargamamos los distintos dataframes
    print('\nCargando dataframes...\n')
    global anime_df, ratings_df, valoraciones_df, movie_df, tv_df, union_ratings_valoraciones_df
    try:    
        # Crear headers para unir los dataframes ratings con valoraciones_EP
        union_headers = StructType([StructField('user_id', IntegerType(), True), StructField('anime_id', IntegerType(), True), StructField('rating', FloatType(), True)])
        anime_df = spark.read.option('encoding', 'UTF-8').csv('data/anime.csv', inferSchema=True, header=True, sep=',')
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
    # Entrenamos el modelo. La estrategia cold start con 'drop' descarata valores NaN en evaluaci??n
    als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="anime_id", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(training)
    # Evaluamos el modelo con RMSE
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions) 
    print("Root-mean-square error = " + str(rmse))
    # Generando los 100 mejores animes para el usuario ID:666666
    print("\nGenerando recomendaciones para el usuario 666.666...\n")
    userRecs = model.recommendForUserSubset(valoraciones_df.filter(valoraciones_df.user_id == 666666), 100)

    # Pasar a dataframe userRecs
    recomendaciones = userRecs.head().recommendations # -> te quedas con la columna recommendations
    recomendaciones_df = spark.createDataFrame(data=recomendaciones) # -> generado dataframe
    
    # Unir recomendaciones para ID:666666 y dataframe peliculas
    recomendaciones_df.createOrReplaceTempView('sqlRecomendaciones')
    movie_df.createOrReplaceTempView('sqlMovies')
    ratings_movies_df = spark.sql(''' SELECT sqlRecomendaciones.anime_id, sqlMovies.Type, sqlMovies.`Name`, sqlMovies.`English name`, sqlMovies.`Japanese name` FROM sqlMovies 
    JOIN sqlRecomendaciones ON sqlMovies.ID = sqlRecomendaciones.anime_id LIMIT 5''')
    
    # Unir recomendaciones para ID:666666 y dataframe series
    tv_df.createOrReplaceTempView('sqlSeries')
    ratings_series_df = spark.sql(''' SELECT sqlRecomendaciones.anime_id, sqlSeries.Type, sqlSeries.`Name`, sqlSeries.`English name`, sqlSeries.`Japanese name` FROM sqlSeries
     JOIN sqlRecomendaciones ON sqlSeries.ID = sqlRecomendaciones.anime_id LIMIT 5''')
    
    # Mostrar/guardar recomendaciones
    print("\nPeliculas recomendadas:")
    ratings_movies_df.show()
    txt_movies = ratings_movies_df.toPandas()
    np.savetxt(r'movies.txt', txt_movies.values, encoding='UTF-8', fmt='%s', delimiter='; ')

    print("\nSeries recomendadas:")
    ratings_series_df.show()
    txt_series = ratings_series_df.toPandas()
    np.savetxt(r'series.txt', txt_series.values, encoding='UTF-8', fmt='%s', delimiter='; ')


if __name__ == '__main__':
    load_data()
    als_recommendation()
