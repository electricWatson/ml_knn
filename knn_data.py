import pandas as pd

tracks = pd.read_csv('tracks.txt', sep='\t')

# Use the tracks danceability, energy, key, and loudness as attributes
X = tracks[['danceability', 'energy', 'key', 'loudness']]
# Use the tracks genre as the thing we are trying to predict
y = tracks['genre']
d = dict([(genre, i) for i, genre in enumerate(set(y))])
# mapping of each genre to an int
y_colors = [d[genre] for genre in y] 

def printAttributeCount(x):
    print("Attribute count: " + str(len(x.columns)))

def printClassDistributions(y):
    dc = dict([(genre, 0) for genre in set(y)])
    for genre in y:
        dc[genre] += 1
    print("Class Distributions: ")
    for genre in dc:
        print("\t" + genre + ": " + str(dc[genre]))