import operator
import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

MAX_ITER = 20

def closest(p, centroids):
  return min([(i, linalg.norm(p - c, norm)) for i, c in enumerate(centroids)], key=operator.itemgetter(1))[0]

conf = SparkConf()
sc = SparkContext(conf=conf)

# Load the data
data = sc.textFile(sys.argv[1]).map(lambda line: np.array([float(x) for x in line.split(' ')])).cache()

# Load the initial centroids c1
centroids1 = sc.textFile(sys.argv[2]).map(lambda line: np.array([float(x) for x in line.split(' ')])).collect()

# Load the initial centroids c2
centroids2 = sc.textFile(sys.argv[3]).map(lambda line: np.array([float(x) for x in line.split(' ')])).collect()

norm = int(sys.argv[4])
cost1 = []
cost2 = []

for _ in range(MAX_ITER):
  # Map each point onto a combo of the point, the closest centroid, and a count=1
  assign1 = data.map(lambda p: (closest(p, centroids1), (p, 1)))
  # Calculate the cost from the assignments
  cost1 += [assign1.map(lambda (c, (p, n)): linalg.norm(p - centroids1[c], norm)).sum()]
  # Average the points for each centroid by summing points and the point counts
  # in the reducer and then mapping to the sum divided by the count
  centroids1 = assign1.reduceByKey(lambda (p1, n1), (p2, n2): (p1 + p2, n1 + n2)).map(lambda (c, (p, n)): p / float(n)).collect()
  # Map each point onto a combo of the point, the closest centroid, and a count=1
  assign2 = data.map(lambda p: (closest(p, centroids2), (p, 1)))
  # Calculate the cost from the assignments
  cost2 += [assign2.map(lambda (c, (p, n)): linalg.norm(p - centroids2[c], norm)).sum()]
  # Average the points for each centroid by summing points and the point counts
  # in the reducer and then mapping to the sum divided by the count
  centroids2 = assign2.reduceByKey(lambda (p1, n1), (p2, n2): (p1 + p2, n1 + n2)).map(lambda (c, (p, n)): p / float(n)).collect()

sc.stop()

x = range(1, MAX_ITER + 1)
fig, ax = plt.subplots()
ax.plot(x, cost1)
ax.plot(x, cost2)
plt.show()