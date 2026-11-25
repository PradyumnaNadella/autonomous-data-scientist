import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from features import build_features
import joblib
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Check for pipeline logic
# The team must have used a pipeline, let's see how they handled the ID/Group/CabinNum columns.
# They didn't show the pipeline code. Let me check if they actually did the preprocessing properly.
# The Top 5 features listed in report: 1. num__VRDeck, 2. num__Age, 3. num__FoodCourt, 4. num__Spa, 5. num__RoomService
# Let's see if those are actually the top features.

train = pd.read_csv('train.csv')
X = build_features(train.drop('Transported', axis=1))
y = train['Transported']

# Determine types
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# 'CabinNum' should probably be numeric?
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group', 'Surname']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', lgb.LGBMClassifier(learning_rate=0.05, n_estimators=100))])

# Run CV
skf = StratifiedKFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
print(f"Computed CV Score: {scores.mean()}")
