import knn_test as kt
import knn_data as kd
# Make KNN with K=3 nearest neighbors, and a minkowski distance metric with p=2 (Euclidean distance)
BASE_NEIGHBOR_COUNT = 3
BASE_DISTANCE_METRIC = 2
BASE_TEST_SIZE = 0.25

kt.showKNNClassificationHeuristics(kd.X, kd.y, kd.y_colors, BASE_NEIGHBOR_COUNT, BASE_DISTANCE_METRIC, BASE_TEST_SIZE)