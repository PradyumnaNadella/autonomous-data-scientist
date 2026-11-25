import pandas as pd
train = pd.read_csv('train.csv')
print(train.columns)
# Inspect the actual usage of OneHotEncoder on high cardinality features
# e.g., 'Cabin' or 'Name' (if it wasn't dropped)
# The code drops 'Name' but not 'Cabin' or 'HomePlanet'.
# 'Cabin' is a string with '/' separators, treating it as categorical will blow up or be noisy.
