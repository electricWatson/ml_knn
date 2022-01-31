import pandas as pd

tracks = pd.read_csv('tracks.txt', sep='\t')

# Use the tracks danceability, energy, key, and loudness as attributes
X = tracks[['danceability', 'energy', 'key', 'loudness']]
# Use the tracks genre as the thing we are trying to predict
y = tracks['genre']
d = dict([(y, x+1) for x,y in enumerate(set(y))])
# mapping of each genre to an int
y_colors = [d[genre] for genre in y] 
