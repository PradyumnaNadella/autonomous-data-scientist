import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('train.csv')
X = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported'].astype(int)

# Identify features
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# Note: 'PassengerId' was not dropped in build_features, need to handle it.
# Actually it exists as a column in the output of build_features.
cat_cols = [c for c in X.columns if c not in num_cols and c != 'PassengerId']

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# CV
lr_pipe = Pipeline([('pre', preprocessor), ('model', LogisticRegression(max_iter=1000))])
rf_pipe = Pipeline([('pre', preprocessor), ('model', RandomForestClassifier(n_estimators=100))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_scores = cross_val_score(lr_pipe, X.drop('PassengerId', axis=1), y, cv=cv, scoring='accuracy')
rf_scores = cross_val_score(rf_pipe, X.drop('PassengerId', axis=1), y, cv=cv, scoring='accuracy')

print(f"LR CV Accuracy: {lr_scores.mean():.4f}")
print(f"RF CV Accuracy: {rf_scores.mean():.4f}")
