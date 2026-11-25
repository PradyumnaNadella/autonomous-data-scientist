import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('train.csv')
X_raw = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported'].astype(int)

# Force mixed types to string
for col in X_raw.columns:
    X_raw[col] = X_raw[col].astype(str)

# Numerical columns were actually strings now, so we need to identify them manually
# Or better: let's convert them to numeric where possible
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in num_cols:
    X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')

# Re-identify types
cat_cols = [c for c in X_raw.columns if c not in num_cols]

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# Test cross-val
model = RandomForestClassifier(n_estimators=50, random_state=42)
pipeline = Pipeline([('pre', preprocessor), ('model', model)])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_raw, y, cv=cv, scoring='accuracy')
print(f"RandomForest: {scores.mean():.4f}")
