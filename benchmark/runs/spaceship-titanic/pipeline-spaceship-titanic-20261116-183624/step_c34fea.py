import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from features import build_features

# Load and prepare
train = pd.read_csv('train.csv')
train = build_features(train)
X = train.drop(columns=['Transported', 'PassengerId'])
y = train['Transported']

# Convert columns with bool to string for OHE compatibility
for col in ['CryoSleep', 'VIP']:
    X[col] = X[col].astype(str)

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'IdInGroup', 'Deck', 'Num', 'Side']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                      ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
])

# Model 1: RF
rf_pipeline = Pipeline([('preprocessor', preprocessor), ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
scores_rf = cross_val_score(rf_pipeline, X, y, cv=5, scoring='accuracy')

# Model 2: XGB
xgb_pipeline = Pipeline([('preprocessor', preprocessor), ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))])
scores_xgb = cross_val_score(xgb_pipeline, X, y, cv=5, scoring='accuracy')

def log_experiment(name, score):
    print(f"EXPERIMENT LOG: {name} - Accuracy: {score:.4f}")

log_experiment("RandomForest", scores_rf.mean())
log_experiment("XGBoost", scores_xgb.mean())

# Submit
rf_pipeline.fit(X, y)
test = pd.read_csv('test.csv')
test_feat = build_features(test)
for col in ['CryoSleep', 'VIP']:
    test_feat[col] = test_feat[col].astype(str)

preds = rf_pipeline.predict(test_feat.drop(columns=['PassengerId']))
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds})
submission.to_csv('predictions.csv', index=False)
