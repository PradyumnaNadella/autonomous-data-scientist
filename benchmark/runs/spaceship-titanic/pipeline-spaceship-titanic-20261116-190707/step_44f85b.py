from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from features import build_features
import pandas as pd

# Load data
train_df = pd.read_csv('train.csv')
X = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported'].astype(int)

# Re-define steps (must re-define as tool processes are fresh)
def cast_to_str(df):
    return df.astype(str)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

cat_pipeline = Pipeline([
    ('cast', FunctionTransformer(cast_to_str)),
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])

# Tune LGBM
lgbm = LGBMClassifier(random_state=42, verbose=-1)
params = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth': [3, 5, 7]
}

clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', lgbm)])
random_search = RandomizedSearchCV(clf, params, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)

random_search.fit(X, y)
print(f"Best Score: {random_search.best_score_:.4f}")
print(f"Best Params: {random_search.best_params_}")
