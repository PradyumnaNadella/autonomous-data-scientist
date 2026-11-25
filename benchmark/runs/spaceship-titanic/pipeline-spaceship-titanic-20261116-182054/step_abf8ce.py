from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

# Prepare data
train_df = pd.read_csv('train.csv')
X_raw = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported'].astype(int)

# Identify numeric correctly from the start
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = [c for c in X_raw.columns if c not in num_cols and c != 'PassengerId']

# Convert columns to appropriate types
for col in cat_cols:
    X_raw[col] = X_raw[col].astype(str)

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# LogisticRegression
pipeline_lr = Pipeline([('pre', preprocessor), ('model', LogisticRegression(max_iter=500))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_lr = cross_val_score(pipeline_lr, X_raw.drop('PassengerId', axis=1), y, cv=cv, scoring='accuracy')
print(f"LogisticRegression: {scores_lr.mean():.4f}")

# RandomForest
pipeline_rf = Pipeline([('pre', preprocessor), ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
scores_rf = cross_val_score(pipeline_rf, X_raw.drop('PassengerId', axis=1), y, cv=cv, scoring='accuracy')
print(f"RandomForest: {scores_rf.mean():.4f}")
