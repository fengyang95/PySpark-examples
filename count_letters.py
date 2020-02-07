# standford cs246 winter 2020 colab 1

from pyspark.sql import *

if __name__=='__main__':

    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext

    documents = sc.textFile('./data/pg100.txt')

    counts = documents.flatMap(lambda line: line.split()).filter(lambda word: word.isalpha()) \
        .map(lambda word: word[0].lower()) \
        .map(lambda letter: (letter, 1)) \
        .reduceByKey(lambda n1, n2: n1 + n2).sortByKey(lambda letter_cnt:letter_cnt[0])

    counts.toDF(['letter','count']).show(26)
    sc.stop()

    """
|letter| count|
+------+------+
|     a| 72155|
|     b| 34065|
|     c| 19618|
|     d| 19492|
|     e| 12666|
|     f| 25624|
|     g| 13682|
|     h| 44569|
|     i| 52986|
|     j|  1474|
|     k|  5861|
|     l| 17256|
|     m| 39937|
|     n| 19183|
|     o| 36408|
|     p| 15838|
|     q|  1299|
|     r|  7973|
|     s| 44741|
|     t|105860|
|     u|  7008|
|     v|  3361|
|     w| 46934|
|     x|     1|
|     y| 20646|
|     z|    32|
+------+------+
    """