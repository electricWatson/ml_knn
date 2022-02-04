import knn_test as kt
import knn_data as kd
import matplotlib.pyplot as plt

# For Python Script.1
kd.printAttributeCount(kd.X)
# For Python Script.2a
kd.printClassDistributions(kd.y)
# For Python Script.2b
kd.plotClassDistributions(kd.X, kd.y_colors, 'danceability', 'energy', 'loudness')
# For Individual Report.3
kd.printClassStatistics(kd.X)

# Make KNN with K=3 nearest neighbors, minkowski distance metric with p=2 (Euclidean distance), test size of 0.25 as defaults
BASE_NEIGHBOR_COUNT = 3
BASE_DISTANCE_METRIC = 2
BASE_TEST_SIZE = 0.25
kt.plotKNNClassificationHeuristics(kd.X, kd.y, kd.y_colors, BASE_NEIGHBOR_COUNT, BASE_DISTANCE_METRIC, BASE_TEST_SIZE)

plt.show()