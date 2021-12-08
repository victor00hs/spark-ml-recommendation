from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.mllib.recommendation import ALS, Rating 

spark = SparkSession.builder.appName("main").master("local[*]").getOrCreate()

anime_df, ratings_df, valoraciones_df, movie_df, tv_df, union_ratings_valoraciones_rdd = None, None, None, None, None, None

def load_data():
    # Cargamamos los distintos dataframes
    print('\nCargando dataframes...\n')
    global anime_df, ratings_df, valoraciones_df, movie_df, tv_df, union_ratings_valoraciones_rdd
    # Crear headers para unir los dataframes ratings con valoraciones_EP
    union_headers = StructType([StructField('user_id', IntegerType(), True), StructField('anime_id', IntegerType(), True), StructField('rating', IntegerType(), True)])
    anime_df = spark.read.option('encoding', 'UTF-8').csv('data/anime.csv', inferSchema=True, header=True, sep=',')
    # Ponemos en cache este dataframe ya que es muy pesado y asi agilizamos las operaciones realizadas en él
    ratings_df = spark.read.option('encoding', 'UTF-8').csv('data/rating_complete.csv', inferSchema=True, header=True, sep=',').cache()
    valoraciones_df = spark.read.option('encoding', 'UTF-8').csv('data/valoraciones_EP.csv', inferSchema=True, schema=union_headers, header=True, sep=',')
    movie_df, tv_df = anime_df.filter(anime_df.Type == 'Movie'), anime_df.filter(anime_df.Type == 'TV')

    # Pide incorporar las valoraciones de EP al fichero de rating_complete
    union_ratings_valoraciones_rdd = ratings_df.union(valoraciones_df).rdd # Se pasa a rdd para poder hacer un map
    print('\nSe han cargado los dataframes...\n')

def als_recommendation():
    print('\nEntrenando modelo de recomendacion...\n')
    rank, numIterations, userID = 10, 6, 666666
    # Explicar porque pasamos a rdd y luego otra vez a dataframe
    ratings_rdd = union_ratings_valoraciones_rdd.map(lambda l: l.split()).map(lambda l: Rating(int(l[0]), int(l[1]), int(l[2])))
    print(ratings_rdd.take(5))
    """ model = ALS.train(ratings_rdd, rank, numIterations)
    print('\nModelo entrenado\n\nLas 5 series recomendadas para el usuario {} son: \n'.format(userID))
    user_recommendations = model.recommendProducts(userID, 5)
    print(user_recommendations) """

if __name__ == '__main__':
    load_data()
    als_recommendation()
