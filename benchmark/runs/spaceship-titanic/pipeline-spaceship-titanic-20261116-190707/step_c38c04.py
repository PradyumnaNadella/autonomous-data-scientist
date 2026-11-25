import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from lightgbm import LGBMClassifier
from features import build_features

def cast_to_str(df):
    return df.astype(str)

train_df = pd.read_csv('train.csv')
X = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported'].astype(int)

# Check what the pipeline actually processes
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# Note: 'Deck', 'Side', 'Group', 'Surname' were added to features but are NOT in the preprocessor list
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

# The preprocessor is missing 'Group' and 'Surname' which were built!
# Are there other missing ones?
# numeric_features in training set:
print("Missing in numerical:", X[numeric_features].isnull().sum())
print("Missing in categorical:", X[categorical_features].isnull().sum())
