from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from features import build_features

# Load and prepare
train = pd.read_csv('train.csv')
train = build_features(train)
X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported']

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'IdInGroup', 'Deck', 'Num', 'Side']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                      ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Model 1: RF
rf_pipeline = Pipeline([('preprocessor', preprocessor), ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
scores_rf = cross_val_score(rf_pipeline, X, y, cv=5, scoring='accuracy')

# Model 2: XGB
xgb_pipeline = Pipeline([('preprocessor', preprocessor), ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))])
scores_xgb = cross_val_score(xgb_pipeline, X, y, cv=5, scoring='accuracy')

print(f"Random Forest CV Accuracy: {scores_rf.mean():.4f}")
print(f"XGBoost CV Accuracy: {scores_xgb.mean():.4f}")

# Helper for logging
def log_experiment(name, score):
    print(f"EXPERIMENT LOG: {name} - Accuracy: {score:.4f}")

log_experiment("RandomForest", scores_rf.mean())
log_experiment("XGBoost", scores_xgb.mean())
