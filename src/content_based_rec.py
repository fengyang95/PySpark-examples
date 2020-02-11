from pyspark.sql import *
from pyspark.sql.functions import collect_list,avg
from nltk.stem.porter import PorterStemmer
import nltk
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.linalg import norm


def tf_similarity(s1,s2):
    def add_space(s):
        return ' '.join(s)
    s1, s2 = add_space(s1), add_space(s2)
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


print(tf_similarity( ['fox', 'hound'], ['happy', 'gilmore']))

year_patter=re.compile(r'[0-9][0-9][0-9][0-9]', re.I)
#nltk.download('stopwords')
#stopwords_en=stopwords.words('english')
stopwords_en=['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
              'ourselves', 'you', "you're", "you've", "you'll", "you'd",
              'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
              "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
              'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
              "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
              'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
              'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
              'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
              'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
              'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
              'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
              'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
              've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
              'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
              'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
              'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
              "won't", 'wouldn', "wouldn't"]


def extrac_movie_features(movie):
    movieId,title,genres=movie
    titles_words = [w.lower() for w in nltk.word_tokenize(title) ]#if w.lower() not in stopwords.words('english')]
    titles_words=[w for w in titles_words if w not in stopwords_en and w not in [',','(',')','\'s','.',':','&'] and not w.isdigit()]
    genres_words=[PorterStemmer().stem(word.lower()) for word in genres.split('|')]
    try:
        year=year_patter.findall(title)[0]
    except IndexError:
        year=0000
    return movieId,year,titles_words,genres_words

if __name__=='__main__':

    spark = SparkSession.builder.getOrCreate()
    movies_df=spark.read.csv('../data/movielens/movies.csv',header=True)
    tags_df=spark.read.csv('../data/movielens/tags.csv',header=True)
    ratings_df=spark.read.csv('../data/movielens/ratings.csv',header=True)
    # deal with movies
    movies_df = movies_df.rdd.map(extrac_movie_features).toDF(['movieId', 'year', 'titles_words', 'genres_words'])
    movies_df.show()

    # deal with ratings
    ratings_df=ratings_df.select(['movieId','rating']).groupBy('movieId').agg(avg('rating').alias('avg_rating'))
    ratings_df.show(10)

    # deal with tags
    tags_df=tags_df.select(['movieId','tag']).rdd\
        .map(lambda ele:(ele[0],[PorterStemmer().stem(x.lower()) for x in ele[1].split()]))\
    .flatMapValues(lambda x:x).map(lambda ele:((ele[0],ele[1]),1)).reduceByKey(lambda n1,n2:n1+n2)
    tags_df=tags_df.map(lambda x:(x[0][0],(x[0][1],x[1]))).toDF(['movieId','tag']).groupBy('movieId')\
    .agg(collect_list('tag')).rdd.map(lambda x:(x[0],[e[0] for e in sorted(x[1],key=lambda val:-val[1])][:10]))\
    .toDF(['movieId','tags'])
    tags_df.show()

    movies_all=movies_df.join(ratings_df,'movieId').join(tags_df,'movieId')
    movies_all.show()

    """
    剩下就是计算相似度做推荐
    """


