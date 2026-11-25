# Log experiments
print("Logging experiments...")
# These are manual calls because the tool is in the environment
# log_experiment(model="RandomForest", cv_score=0.7962, notes="Baseline")
# log_experiment(model="XGBoost", cv_score=0.7923, notes="Baseline")
# log_experiment(model="LightGBM", cv_score=0.8067, notes="Baseline")

# Let's tune LightGBM
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
import pandas as pd
from features import build_features

train_df = pd.read_csv('train.csv')
X = build_features(train_df.drop(columns=['Transported']))
y = train_df['Transported'].astype(int)
for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'GroupId', 'Deck', 'Side']:
    X[col] = X[col].astype(str)

num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum', 'TotalSpent', 'HasSpent']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'GroupId', 'Deck', 'Side']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), 
                      ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

pipe = Pipeline([('preprocessor', preprocessor), ('model', LGBMClassifier(random_state=42, verbose=-1))])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [5, 10]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print(f"Best score: {grid.best_score_:.4f}")
print(f"Best params: {grid.best_params_}")
