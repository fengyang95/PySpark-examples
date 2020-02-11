# standford cs246 winter 2020 hw2 q2
# implementate kmeans using pyspark

from pyspark.sql import SparkSession
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(p1,p2):
    return float((np.mean((p1-p2)**2)))

def manhattan_distance(p1,p2):
    return float(np.mean(np.abs(p1-p2)))

def line2point(line):
    return np.array([float(num) for num in line.split(" ")])

def get_init_centers(txt_file_path):
    centers=np.loadtxt(txt_file_path)
    centers_map={}
    for cluster in range(len(centers)):
        centers_map[cluster]=centers[cluster]
    return centers_map

def assigned_centers(point,centers,distance_func):
    ans= min(
        [(cluster, point, distance_func(point, centers[cluster])) for cluster in centers],
        key=lambda tup: tup[2])
    return ans

class KMeans:
    def __init__(self,max_iter,init_centers,distance_func):
        self.max_iter=max_iter
        self.init_centers=init_centers
        self.distance_func=distance_func
        self.costs_=[]
        self.centers=None

    def fit(self,dataset):
        current_centers=self.init_centers
        for _ in range(self.max_iter):
            assigned_center_dataset=dataset.map(lambda point:assigned_centers(point,current_centers,self.distance_func))

            self.costs_.append(assigned_center_dataset.map(lambda tup:tup[2]).sum())

            centers_count=assigned_center_dataset.map(lambda tup:(tup[0],1)).reduceByKey(lambda c1,c2:c1+c2).collectAsMap()
            current_centers=assigned_center_dataset.map(lambda tup:(tup[0],tup[1])).reduceByKey(lambda p1,p2:p1+p2)\
                            .map(lambda tup:(tup[0],tup[1]/centers_count[tup[0]])).collectAsMap()
        self.centers=current_centers


    def predict(self,dataset):
        pass


if __name__ == "__main__":

    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    dataset = sc.textFile("../data/data.txt")
    dataset=dataset.map(line2point)
    init_centers_1 = get_init_centers("../data/c1.txt")
    init_centers_2 = get_init_centers("../data/c2.txt")

    # euclidean c1
    kmeans_1=KMeans(max_iter=20,init_centers=init_centers_1,distance_func=euclidean_distance)
    kmeans_1.fit(dataset)

    # euclidean c2
    kmeans_2=KMeans(max_iter=20,init_centers=init_centers_2,distance_func=euclidean_distance)
    kmeans_2.fit(dataset)

    import matplotlib.pyplot as plt

    plt.plot(range(len(kmeans_1.costs_)), kmeans_1.costs_, label='c1')
    plt.plot(range(len(kmeans_2.costs_)), kmeans_2.costs_, label='c2')
    plt.title('euclidean')
    plt.legend()
    plt.show()

    # manhattan c1
    kmeans_3=KMeans(max_iter=20,init_centers=init_centers_1,distance_func=manhattan_distance)
    kmeans_3.fit(dataset)

    # manhattan c2
    kmeans_4=KMeans(max_iter=20,init_centers=init_centers_2,distance_func=manhattan_distance)
    kmeans_4.fit(dataset)

    plt.plot(range(len(kmeans_3.costs_)), kmeans_3.costs_, label='c1')
    plt.plot(range(len(kmeans_4.costs_)), kmeans_4.costs_, label='c2')
    plt.title('manhattan')
    plt.legend()
    plt.show()


    sc.stop()
