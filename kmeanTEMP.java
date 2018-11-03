import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class kmeanTEMP {

	private static int K = 10, DIM = 20;
	private static double THR = 0.01;
	private static ArrayList<List<Double>> centroid = new ArrayList<List<Double>>();
	
	public static void main(String[] args) throws Exception {
		
		JavaSparkContext sc = new JavaSparkContext("local[*]", "programname", System.getenv("SPARK_HOME"), System.getenv("JARS"));

		// Read the points
	    JavaRDD<String> data1 = sc.textFile(args[0]);
	    JavaRDD<List<Double>> points = data1.map(d -> getPoints(d));
	    
	    // Read the centroids
	    List<String> data2 = sc.textFile(args[1]).collect();
	    for (int i=0; i<K; i++)
			centroid.add(getPoints(data2.get(i)));
	    
	    ArrayList<List<Double>> updateCentroid = (ArrayList<List<Double>>) centroid.clone();
	    
	    do {
	    	
	    	centroid = (ArrayList<List<Double>>) updateCentroid.clone();
	    	
	    	// TODO Assign points
	    	
	    	// TODO Update centroids
	    	
	    } while (diff(centroid, updateCentroid)>THR);
	    	
	    
	    for (int i=0; i<K; i++)
	    	System.out.println(centroid.get(i));
	}
	
	private static double diff(ArrayList<List<Double>> c1, ArrayList<List<Double>> c2) {
		// TODO Compute the sum of distances for each pair of centroids, one from c1 and the other from c2
		return 0;
	}

	private static Tuple2<Integer, List<Double>> update(Tuple2<Integer, Iterable<List<Double>>> c) {
		// c.1 is the ID of the centroid. c.2 is the list of all the points assigned to the centroid.
		// TODO Compute the average of all assigned points to update the centroid.
		return null;
	}

	private static Tuple2<Integer, List<Double>> nearestC(List<Double> p) {
		// p is one point
		// TODO Find the nearest centroid to p, and produce the tuple (centroidID, p)
		return null;
	}
	
	private static double dist(List<Double> p, List<Double> q) {
		// Compute the Euclidean distance between two points p and q
		double distance = 0.0;
		
		for (int i=0; i<DIM; i++)
			distance += (p.get(i)-q.get(i))*(p.get(i)-q.get(i));
		return Math.sqrt(distance);
	}

	private static List<Double> getPoints(String d) {
		String[] s_point = d.split("\t");
		Double[] d_point = new Double[DIM];
		for (int i=0; i<DIM; i++)
			d_point[i] = Double.parseDouble(s_point[i]);
		return Arrays.asList(d_point);
	}
}