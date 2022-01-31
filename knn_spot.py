from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# Make KNN with K=3 (3 nearest neighbors), and P=2 (Euclidean distance (Minkowski distance p=2))
knn = KNeighborsClassifier(n_neighbors=3, p=2)

tracks = pd.read_csv('tracks.txt', sep='\t')

# Use the tracks danceability, energy, key, and loudness as attributes
X = tracks[['danceability', 'energy', 'key', 'loudness']]
# Use the tracks genre as the thing we are trying to predict
y = tracks['genre']
d = dict([(y, x+1) for x,y in enumerate(set(y))])
# mapping of each genre to an int
genre_ints = [d[genre] for genre in y]

knn.fit(X, y)

def knn_predict(knn, danceability, energy, key, loudness):
    unknown = pd.DataFrame([[danceability, energy, key, loudness]], columns=['danceability', 'energy', 'key', 'loudness'])
    genre_prediction = knn.predict(unknown)
    print(genre_prediction[0])
    print(knn.predict_proba(unknown))

print("Predictions:")
# hiphop || Lonely (with Lil Wayne) - DaBaby, Lil Wayne
knn_predict(knn, 0.718, 0.628, 0, -5.334)
# doom || Rat King - Code
knn_predict(knn, 0.346, 0.938, 6, -9.088)

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# Train the classifier (fit the estimator) using the training data
knn.fit(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
knn.score(X_test, y_test)

# How sensitive is k-NN classification accuracy to the choice of the 'k' parameter?
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

# visualization
# plotting a scatter matrix
from matplotlib import cm
from pandas.plotting import scatter_matrix
cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X, c=genre_ints, marker='o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

# plotting a 3D scatter plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d   # must keep
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X['danceability'], X['energy'], X['loudness'], c = genre_ints, marker = 'o', s=100)
ax.set_xlabel('danceability')
ax.set_ylabel('energy')
ax.set_zlabel('loudness')

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20])

# How sensitive is k-NN classification accuracy to the train/test split proportion?
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
knn = KNeighborsClassifier(n_neighbors=5)
plt.figure()
for s in t:
    scores = []
    for i in range(1, 1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')
plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy')
plt.show()
