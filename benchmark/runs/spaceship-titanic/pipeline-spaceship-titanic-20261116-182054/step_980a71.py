from sklearn.linear_model import LogisticRegression
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

for col in X_raw.columns:
    X_raw[col] = X_raw[col].astype(str)

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in num_cols:
    X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')

cat_cols = [c for c in X_raw.columns if c not in num_cols]

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Compare LogReg
pipeline_lr = Pipeline([('pre', preprocessor), ('model', LogisticRegression(max_iter=1000))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_lr = cross_val_score(pipeline_lr, X_raw, y, cv=cv, scoring='accuracy')
print(f"LogisticRegression: {scores_lr.mean():.4f}")
