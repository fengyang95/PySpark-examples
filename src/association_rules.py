# standford cs246 hw1 q2 d and e
from pyspark.sql import SparkSession
from functools import partial
from itertools import combinations

if __name__=='__main__':
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    lines=sc.textFile('../data/browsing.txt')

    frequent_items=lines.flatMap(lambda l:l.split()).map(lambda ele:(ele,1)).\
        reduceByKey(lambda e1,e2:e1+e2).filter(lambda x:x[1]>=100)

    frequent_itemset=frequent_items.collectAsMap()

    frequent_pairs = lines.map(lambda l: l.split()).flatMap(partial(combinations, r=2)).map(lambda pair: sorted(pair)) \
        .map(lambda pair: (tuple(pair), 1)).filter(
        lambda ele: ele[0][0] in frequent_itemset and ele[0][1] in frequent_itemset).reduceByKey(
        lambda p1, p2: p1 + p2).filter(lambda x: x[1] >= 100)

    freq_pairs_count=frequent_pairs.collectAsMap()

    frequent_pairs=frequent_pairs.flatMap(lambda ele:[((ele[0][0],ele[0][1]),ele[1]),((ele[0][1],ele[0][0]),ele[1])])

    frequent_pair_conf=frequent_pairs.map(lambda ele:(ele[0],float(ele[1]/frequent_itemset[ele[0][0]]))).sortBy(lambda x:-x[1])

    frequent_pair_conf.toDF().show(10)
    """
    |[DAI93865, FRO40251]|               1.0|
    |[GRO85051, FRO40251]| 0.999176276771005|
    |[GRO38636, FRO40251]|0.9906542056074766|
    |[ELE12951, FRO40251]|0.9905660377358491|
    |[DAI88079, FRO40251]|0.9867256637168141|
    |[FRO92469, FRO40251]| 0.983510011778563|
    |[DAI43868, SNA82528]| 0.972972972972973|
    |[DAI23334, DAI62779]|0.9545454545454546|
    |[ELE92920, DAI62779]|0.7326649958228906|
    |[DAI53152, FRO40251]| 0.717948717948718|
    """

    triples=lines.map(lambda l:l.split()).flatMap(partial(combinations,r=3)).map(lambda triple:tuple(list(sorted(triple))))

    triples=triples.map(lambda triple:(triple,1)).reduceByKey(lambda t1,t2:t1+t2).filter(lambda x:x[1]>=100)

    freq_triples_conf=triples.flatMap(lambda ele:[(((ele[0][0],ele[0][1]),ele[0][2]),ele[1]),
                                                  (((ele[0][0],ele[0][2]),ele[0][1]),ele[1]),
                                                  (((ele[0][1],ele[0][2]),ele[0][0]),ele[1])])\
    .reduceByKey(lambda t1,t2:t1+t2).map(lambda ele:(ele[0],ele[1]/freq_pairs_count[ele[0][0]])).sortBy(lambda x:(-x[1],x[0][0],x[0][1]))\
                        .map(lambda ele:(ele[0][0],ele[0][1],ele[1]))

    freq_triples_conf.toDF().show(10)
    """
    |[DAI23334, ELE92920]|DAI62779|1.0|
    |[DAI31081, GRO85051]|FRO40251|1.0|
    |[DAI55911, GRO85051]|FRO40251|1.0|
    |[DAI62779, DAI88079]|FRO40251|1.0|
    |[DAI75645, GRO85051]|FRO40251|1.0|
    |[ELE17451, GRO85051]|FRO40251|1.0|
    |[ELE20847, FRO92469]|FRO40251|1.0|
    |[ELE20847, GRO85051]|FRO40251|1.0|
    |[ELE26917, GRO85051]|FRO40251|1.0|
    |[FRO53271, GRO85051]|FRO40251|1.0|
    """
    sc.stop()

