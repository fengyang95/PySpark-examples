# standford cs246 winter 2020 hw1 q1

from pyspark.sql import SparkSession

def line2dataset(line):
    src, dst_line= line.split('\t')
    src = int(src.strip())
    dst_list = [int(x.strip()) for x in dst_line.split(',') if x != '']
    return src, dst_list

def filter_pairs(x):
    if (x[0][0] != x[1][0]) and (not x[0][0] in x[1][1]) and (not x[1][0] in x[0][1]):
        shared = len(list(set(x[0][1]).intersection(set(x[1][1]))))
        return (x[0][0],[x[1][0],shared])

def map_finaldataset(elem):
    src = elem[0]
    dst_commons = elem[1]
    dst_commons=sorted(dst_commons,key=lambda x:(-x[1],x[0]))[:10]
    recommendations=[pair[0] for pair in dst_commons]
    return (src, recommendations)

if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    dataset = sc.textFile("../data/soc-LiveJournal1Adj.txt")

    dataset = dataset.map(line2dataset)

    check_users = [924, 8941, 8942, 9019, 9020, 9021, 9022, 9990, 9992, 9993]
    cartesian = dataset.cartesian(dataset).filter(lambda x: x[0][0] in check_users)

    dataset = cartesian.map(filter_pairs).filter(lambda x: x != None and x[1][1] > 0)\
        .filter(lambda x: x[0] in check_users) \
        .groupByKey().mapValues(list).map(map_finaldataset)

    id_check_dataset = dataset.filter(lambda x: x[0] in check_users).collect()

    for key, val in id_check_dataset:
        print('id:', key,' recommendations:', val)
    sc.stop()
    """
    id: 9020  recommendations: [9021, 9016, 9017, 9022, 317, 9023]
    id: 924  recommendations: [439, 2409, 6995, 11860, 15416, 43748, 45881]
    id: 9992  recommendations: [9987, 9989, 35667, 9991]
    id: 9021  recommendations: [9020, 9016, 9017, 9022, 317, 9023]
    id: 9993  recommendations: [9991, 13134, 13478, 13877, 34299, 34485, 34642, 37941]
    id: 8941  recommendations: [8943, 8944, 8940]
    id: 9022  recommendations: [9019, 9020, 9021, 317, 9016, 9017, 9023]
    id: 9990  recommendations: [13134, 13478, 13877, 34299, 34485, 34642, 37941]
    id: 8942  recommendations: [8939, 8940, 8943, 8944]
    id: 9019  recommendations: [9022, 317, 9023]
    """
