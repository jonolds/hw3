package edu.stanford.cs246

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.SparkContext._
import breeze.linalg.{norm, squaredDistance, DenseVector}

object KMeansScala {
  val MAX_ITER = 20

  // Helper method to find the closest centroid to a point using the given norm
  def closest(p: DenseVector[Double], centroids: Array[DenseVector[Double]], n: Int) : Integer = {
    var best = 0
    var bestD = Double.PositiveInfinity
    
    for (i <- 0 until centroids.length) {
      val dist = norm(p - centroids(i), n)

      if (dist < bestD) {
        best = i
        bestD = dist
      }
    }

    return best
  }

  def main(args: Array[String]) {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    // Load the data
    val data = sc.textFile(args(0))
        .map(line => DenseVector(line.split(" ").map(coord => coord.toDouble))).cache

    // Load the initial centroids c1
    var centroids1 = sc.textFile(args(1))
        .map(line => DenseVector(line.split(" ").map(coord => coord.toDouble))).collect

    // Load the initial centroids c2
    var centroids2 = sc.textFile(args(2))
        .map(line => DenseVector(line.split(" ").map(coord => coord.toDouble))).collect

    val n = args(3).toInt
    var cost1 = Vector[Double]()
    var cost2 = Vector[Double]()

    for (i <- 1 to MAX_ITER) {
      // Map each point onto a combo of the point, the closest centroid, and a count=1
      val assign1 = data.map(p => (closest(p, centroids1, n), (p, 1)))
      // Calculate the cost from the assignments
      cost1 = cost1 :+ assign1.map{case (c, (p, n)) => norm(p - centroids1(c), n)}.sum
      // Average the points for each centroid by summing points and the point counts
      // in the reducer and then mapping to the sum divided by the count
      centroids1 = assign1.reduceByKey{case ((p1, n1), (p2, n2)) => (p1 + p2, n1 + n2)}
          .map{case (c, (p, n)) => p / n.toDouble}.collect
      // Map each point onto a combo of the point, the closest centroid, and a count=1
      val assign2 = data.map(p => (closest(p, centroids2, n), (p, 1)))
      // Calculate the cost from the assignments
      cost2 = cost2 :+ assign2.map{case (c, (p, n)) => norm(p - centroids2(c), n)}.sum
      // Average the points for each centroid by summing points and the point counts
      // in the reducer and then mapping to the sum divided by the count
      centroids2 = assign2.reduceByKey{case ((p1, n1), (p2, n2)) => (p1 + p2, n1 + n2)}
          .map{case (c, (p, n)) => p / n.toDouble}.collect
    }

    sc.stop()

    // Print out the cost vectors
    println(cost1)
    println(cost2)
  }
}
