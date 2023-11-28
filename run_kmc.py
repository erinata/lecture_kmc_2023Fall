import pandas
import matplotlib.pyplot as pyplot
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score 

dataset = pandas.read_csv("dataset.csv")

print(dataset)

dataset = dataset.values

pyplot.scatter(dataset[:,0],dataset[:,1])
pyplot.savefig("scatterplot.png")
pyplot.close()


def run_kmeans(n, dataset):
  machine = KMeans(n_clusters=n)
  machine.fit(dataset)
  results = machine.predict(dataset)
  pyplot.scatter(dataset[:,0],dataset[:,1], c=results)
  pyplot.savefig("scatterplot_kmeans_" + str(n) + ".png")
  pyplot.close()
  print(silhouette_score(dataset, results, metric="euclidean"))


run_kmeans(5, dataset)











