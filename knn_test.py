from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Fits the data {iters} amount of times with each neighbor count, getting the total score of each neighbor count and returning the highest one
def getBestNeighborCount(X, y, distance_metric, test_size, iters):
    bestNeighborCount = -1
    highestAccuracy = 0
    for k in range(1, 10):
        total = 0
        for i in range(iters):
            knn = KNeighborsClassifier(n_neighbors=k, p=distance_metric)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            knn.fit(X_train, y_train)
            total += knn.score(X_test, y_test)
        if total > highestAccuracy:
            highestAccuracy = total
            bestNeighborCount = k
    return bestNeighborCount

# Fits the data {iters} amount of times with each distance metric, getting the total score of each distance metric and returning the highest one
def getBestDistanceMetric(X, y, neighbor_count, test_size, iters):
    bestDistanceMetric = -1
    highestAccuracy = 0
    for d in np.arange(1, 5, 0.5):
        total = 0
        for i in range(iters):
            knn = KNeighborsClassifier(n_neighbors=neighbor_count, p=d)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            knn.fit(X_train, y_train)
            total += knn.score(X_test, y_test)
        if total > highestAccuracy:
            highestAccuracy = total
            bestDistanceMetric = d
    return bestDistanceMetric

# Fits the data {iters} amount of times with each test size, getting the total score of each test size and returning the highest one
def getBestTestSize(X, y, neighbor_count, distance_metric, iters):
    bestTestSize = -1
    highestAccuracy = 0
    knn = KNeighborsClassifier(n_neighbors=neighbor_count, p=distance_metric)
    for ts in np.arange(0.1, 0.9, 0.05):
        total = 0
        for i in range(iters):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts)
            knn.fit(X_train, y_train)
            total += knn.score(X_test, y_test)
        if total > highestAccuracy:
            highestAccuracy = total
            bestTestSize = ts
    return bestTestSize

# plotBest (METRIC) cool documentation notes
# CONFUSION MATRIX: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html for Individual Report.5

# Sets up the statistically best heuristic values for the KNN (k, p, & test size), then displays those as plots
def plotKNNClassificationHeuristics(X, y, y_colors, base_neighbor_count, base_distance_metric, base_test_size):
    best_neighbor_count = base_neighbor_count
    best_distance_metric = base_distance_metric
    best_test_size = base_test_size
    ITERS = 100
    for iterations in range(1,4):
        best_neighbor_count = getBestNeighborCount(X, y, best_distance_metric, best_test_size, ITERS)
        best_distance_metric = getBestDistanceMetric(X, y, best_neighbor_count, best_test_size, ITERS)
        best_test_size = getBestTestSize(X, y, best_neighbor_count, best_distance_metric, ITERS)

    # For Individual Report.5
    # best_neighbor_count = plotBestNeighborCounts(X, y, y_colors, best_distance_metric, best_test_size, ITERS)
    # best_distance_metric = plotBestDistanceMetrics(X, y, y_colors, best_neighbor_count, best_test_size, ITERS)
    # best_test_size = plotBestTestSizes(X, y, y_colors, best_neighbor_count, best_distance_metric, ITERS)

    # For Python Script.3
    print("Test size used: " + str(best_test_size))
    # For Python Script.4
    print("Distance metric used: " + str(best_distance_metric))

    # For Python Script.5
    knn = KNeighborsClassifier(n_neighbors=best_neighbor_count, p=best_distance_metric)
    # plotKnnAccuracy(X, y, y_colors, knn, best_test_size)

    print("best neighbor count " + str(best_neighbor_count))
    print("best distance metric " + str(best_distance_metric))
    print("best test size " + str(best_test_size))