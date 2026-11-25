import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
X = build_features(train.drop('Transported', axis=1))

# Check data types
print("Types:")
print(X[ ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side'] ].dtypes)

# Convert all categorical to string
X[['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']] = X[['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']].astype(str)
print("Updated Types:")
print(X[ ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side'] ].dtypes)
