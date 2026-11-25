import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('train.csv')
X = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported']

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = [c for c in X.columns if c not in num_cols and c != 'PassengerId']

# Convert columns to string for the encoder
X[cat_cols] = X[cat_cols].astype(str)

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Re-run CV
model = Pipeline([('pre', preprocessor), ('model', LogisticRegression(max_iter=1000))])
scores = cross_val_score(model, X.drop('PassengerId', axis=1), y, cv=5, scoring='accuracy')
print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean()}")
