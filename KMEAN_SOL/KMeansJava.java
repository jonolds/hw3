package edu.stanford.cs246;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

public class KMeansJava {
	private static final int MAX_ITER = 20;

	// Helper method to find the closest centroid to a point using the given norm
	private static int closest(List<Double> p, List<List<Double>> centroids, int norm) {
		int best = 0;
		double bestD = Double.POSITIVE_INFINITY;

		for (int i = 0; i < centroids.size(); i++) {
			double dist = norm(p, centroids.get(i), norm);
			if (dist < bestD) {
				best = i;
				bestD = dist;
			}
		}
		return best;
	}

	// Helper function to calulate the norm1 or norm2 of a list of doubles
	private static double norm(List<Double> a1, List<Double> a2, int norm) {
		double dist = 0.0;
		for (int i = 0; i < a1.size(); i++)
			dist += (norm == 2) ? Math.pow(a1.get(i) - a2.get(i), norm) : Math.abs(a1.get(i)- 2.get(i));
		return dist;
	}

	// Holper function to add to lists of doubles elementwise
	private static List<Double> sum(List<Double> a1, List<Double> a2) {
		List<Double> ret = new ArrayList(a1.size());
		for (int i = 0; i < a1.size(); i++)
			ret.add(a1.get(i) + a2.get(i));
		return ret;
	}

	// Helper function to divide all elements in a list by a double
	private static List<Double> div(List<Double> a, double n) {
		List<Double> ret = new ArrayList(a.size());
		for (int i = 0; i < a.size(); i++)
			ret.add(a.get(i)	/ n);
		return ret;
	}

	public static void main(String[] args) throws NumberFormatException {
		SparkConf conf = new SparkConf();
		JavaSparkContext sc = new JavaSparkContext(conf);

		// Load the data
		JavaRDD<List<Double>> data = sc.textFile(args[0])
				.map(line -> Arrays.asList(line.split(" ")).stream().map(Double::valueOf).collect(Collectors.toList())).cache();

		// Load the initial centroids c1
		List<List<Double>> centroids1 = sc.textFile(args[1])
				.map(line -> Arrays.asList(line.split(" ")).stream().map(Double::valueOf).collect(Collectors.toList())).collect();

		// Load the initial centroids c2
		List<List<Double>> centroids2 = sc.textFile(args[2])
				.map(line -> Arrays.asList(line.split(" ")).stream().map(Double::valueOf).collect(Collectors.toList())).collect();

		final int n = Integer.parseInt(args[3]);
		List<Double> cost1 = new ArrayList<>();
		List<Double> cost2 = new ArrayList<>();

		for (int i = 0; i < MAX_ITER; i++) {
			centroids1 = doIteration(centroids1, data, n, cost1);
			centroids2 = doIteration(centroids2, data, n, cost2);
		}
		sc.stop();

		// Print out the cost vectors
		System.out.println(cost1);
		System.out.println(cost2);
	}

	private static List<List<Double>> doIteration(final List<List<Double>> centroids,
		JavaRDD<List<Double>> data, final int n, List<Double> cost1) {
		// Map each point onto a combo of the point, the closest centroid, and a count=1
		JavaPairRDD<Integer, Tuple2<List<Double>, Integer>> assign1 = data.mapToPair(p ->
				new Tuple2<Integer, Tuple2<List<Double>, Integer>>(closest(p, centroids, n), new Tuple2<>(p, 1)));
		// Calculate the cost from the assignments
		JavaDoubleRDD costs = assign1.mapToDouble(p -> norm(p._2._1, centroids.get(p._1), n));

		// If this is norm2, we have to take the sqrt to get the actual cost
		if (n == 2)
			costs = costs.mapToDouble(c -> Math.sqrt(c));

		cost1.add(costs.sum());
		// Average the points for each centroid by summing points and the point counts
		// in the reducer and then mapping to the sum divided by the count
		return assign1.reduceByKey((t1, t2) -> new Tuple2<>(sum(t1._1, t2._1), t1._2 + t2._2))
				.map(t -> div(t._2._1, t._2._2)).collect();
	}
}