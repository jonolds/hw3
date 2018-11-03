import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.SparkSession;

import scala.Tuple2;

public class Stanford {
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
			dist += (norm == 2) ? Math.pow(a1.get(i) - a2.get(i), norm) : Math.abs(a1.get(i) - a2.get(i));
		return dist;
	}

	// Holper function to add to lists of doubles elementwise
	private static List<Double> sum(List<Double> a1, List<Double> a2) {
		List<Double> ret = new ArrayList<>(a1.size());
		for (int i = 0; i < a1.size(); i++)
			ret.add(a1.get(i) + a2.get(i));
		return ret;
	}

	// Helper function to divide all elements in a list by a double
	private static List<Double> div(List<Double> a, double n) {
		List<Double> ret = new ArrayList<>(a.size());
		for (int i = 0; i < a.size(); i++)
			ret.add(a.get(i)	/ n);
		return ret;
	}

	public static void run(SparkSession ss) throws NumberFormatException {

		// Load the data
		JavaRDD<List<Double>> data = ss.read().textFile("data.txt").javaRDD().map(line -> Arrays.asList(line.split("\t")).stream().map(Double::valueOf).collect(Collectors.toList())).cache();

		// Load the initial centroids c1
		List<List<Double>> centroids1 = ss.read().textFile("centroid.txt").javaRDD().map(line -> Arrays.asList(line.split("\t")).stream().map(Double::valueOf).collect(Collectors.toList())).collect();

		List<Double> cost1 = new ArrayList<>();

		for (int i = 0; i < MAX_ITER; i++)
			centroids1 = doIteration(centroids1, data, 1, cost1);

		System.out.println(cost1);
	}

	private static List<List<Double>> doIteration(final List<List<Double>> centroids, JavaRDD<List<Double>> data, final int n, List<Double> cost1) {
		// Map each point onto a combo of the point, the closest centroid, and a count=1
		JavaPairRDD<Integer, Tuple2<List<Double>, Integer>> assign1 = data.mapToPair(p ->new Tuple2<Integer, Tuple2<List<Double>, Integer>>(closest(p, centroids, n), new Tuple2<>(p, 1)));

		// Calculate the cost from the assignments
		JavaDoubleRDD costs = assign1.mapToDouble(p -> norm(p._2._1, centroids.get(p._1), n));

		// If this is norm2, we have to take the sqrt to get the actual cost
		if (n == 2)
			costs = costs.mapToDouble(c -> Math.sqrt(c));
		cost1.add(costs.sum());

		// Average the points for each centroid by summing points and the point counts
		// in the reducer and then mapping to the sum divided by the count
		return assign1.reduceByKey((t1, t2) -> new Tuple2<>(sum(t1._1, t2._1), t1._2 + t2._2)).map(t -> div(t._2._1, t._2._2)).collect();
	}


	/* Main / Standard Setup */
	public static void main(String[] args) throws Exception {
		SparkSession ss = settings();
		run(ss);
		Thread.sleep(20000);
		ss.close();
	}

	static SparkSession settings() throws IOException {
		Logger.getLogger("org").setLevel(Level.WARN);
		Logger.getLogger("akka").setLevel(Level.WARN);
		SparkSession.clearActiveSession();
		SparkSession spark = SparkSession.builder().appName("Kmeans").config("spark.master", "local").config("spark.eventlog.enabled","true").config("spark.executor.cores", "2").getOrCreate();
		SparkContext sc = spark.sparkContext();
		sc.setLogLevel("WARN");
		FileUtils.deleteDirectory(new File("output"));
		return spark;
	}
}
