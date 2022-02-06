import pandas as pd
import matplotlib.pyplot as plt

tracks = pd.read_csv('tracks.txt', sep='\t')

# Use the tracks danceability, energy, key, and loudness as attributes
X = tracks[['danceability', 'energy', 'key', 'loudness']]
# Use the tracks genre as the thing we are trying to predict
y = tracks['genre']
d = dict([(genre, i) for i, genre in enumerate(set(y))])
# mapping of each genre to an int
y_colors = [d[genre] for genre in y] 

def plotClassDistributions(fignum, x, colors, p1, p2, p3):
    fig = plt.figure(fignum)
    subplot = fig.add_subplot(projection = '3d')
    subplot.scatter(x[p1], x[p2], x[p3], c=colors, marker = 'o', s=100)
    subplot.set_xlabel(p1)
    subplot.set_ylabel(p2)
    subplot.set_zlabel(p3)

def printAttributeCount(x):
    print("Attribute count: " + str(len(x.columns)))

def printClassDistributions(y):
    dc = dict([(genre, 0) for genre in set(y)])
    for genre in y:
        dc[genre] += 1
    print("Class Distributions: ")
    for genre in dc:
        print("\t" + genre + ": " + str(dc[genre]))

def printClassStatistics(data):
    print("Minimums attribute values: ")
    print(data.min())
    print("Maximum attribute values: ")
    print(data.max())
    print("Mean attribute values: ")
    print(data.mean())
    print("Median attribute values: ")
    print(data.median())
    print("Mode attribute values: ")
    print(data.mode())
    print("Std dev attribute values: ")
    print(data.std())
