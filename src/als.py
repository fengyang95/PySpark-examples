# -*- coding: utf-8 -*-
# @Time    : 2020/2/11 20:19
# @Author  : Yuanpeng Li
# @FileName: als.py

from pyspark.sql import *
from pyspark.mllib.recommendation import ALS,Rating
import math

def compute_rmse(model,data,n):
    predictions=model.predictAll(data.map(lambda x:(x.user,x.product)))
    data_tmp=data.map(lambda x:((x.user,x.product),x.rating))
    predictions_and_ratings=predictions.map(lambda x:((x.user,x.product),x.rating)).join(data_tmp).values()
    return math.sqrt(predictions_and_ratings.map(lambda x:(x[0]-x[1])**2).reduce(lambda n1,n2:n1+n2)/n)

if __name__=='__main__':

    spark = SparkSession.builder.getOrCreate()
    ratings_df=spark.read.csv('../data/movielens/ratings.csv',header=True)
    movies_df=spark.read.csv('../data/movielens/movies.csv',header=True).select(['movieId','title'])
    #
    ratings=ratings_df.rdd.map(lambda f:
                               (int(f[3])%10,Rating(int(f[0]),int(f[1]),float(f[2]))))
    personal_rating_data=ratings_df.where('userId=1').rdd.map(
        lambda f:Rating(int(f[0]),int(f[1]),float(f[2]))
    )
    num_partions=1
    training=ratings.filter(lambda x:x[0]<6).values().union(personal_rating_data).repartition(num_partions).persist()
    validation=ratings.filter(lambda x:x[0]>=6 and x[0]<8).values()\
        .repartition(num_partions).persist()
    num_validation=validation.count()
    test=ratings.filter(lambda x:x[0]>=8).values().persist()

    count=0
    rank=2
    lambda_=0.01
    num_iter=5

    model = ALS.train(training, rank, num_iter, lambda_)
    validation_rmse = compute_rmse(model, validation, num_validation)
    print('validation_rmse:',validation_rmse)
    # rec
    my_rated_movie_ids=set(personal_rating_data.map(lambda f:f.product).collect())
    candidates=movies_df.rdd.keys().filter(lambda x:x not in my_rated_movie_ids)

    user_id=1
    cand_rdd=candidates.map(lambda x:(user_id,x))

    recommendations=model.predictAll(cand_rdd).sortBy(lambda x:-x.rating).toDF(['user','movieId','rating'])
    recommendations=recommendations.join(movies_df,'movieId')
    recommendations.show()


