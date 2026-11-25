from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from features import build_features

train = pd.read_csv('train.csv')
train_df = build_features(train)

cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side']
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum']

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_features),
    ('cat', Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features)
])

# Random Forest
model_rf = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
scores_rf = cross_val_score(model_rf, train_df.drop(columns=['PassengerId', 'Transported']), train_df['Transported'], cv=5)
print(f"RandomForest CV Accuracy: {scores_rf.mean():.4f}")

# Gradient Boosting
model_gb = Pipeline([('pre', preprocessor), ('clf', GradientBoostingClassifier(random_state=42))])
scores_gb = cross_val_score(model_gb, train_df.drop(columns=['PassengerId', 'Transported']), train_df['Transported'], cv=5)
print(f"GradientBoosting CV Accuracy: {scores_gb.mean():.4f}")
