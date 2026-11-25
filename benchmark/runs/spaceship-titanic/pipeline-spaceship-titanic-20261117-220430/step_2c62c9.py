import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature engineering
X = build_features(train.drop(columns=['Transported']))
y = train['Transported'].astype(int)
X_test = build_features(test)

# Identify features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Define models
rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42))])
xgb = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])

# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model, name in [(rf, 'Random Forest'), (xgb, 'XGBoost')]:
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"{name} CV Score: {scores.mean():.4f}")
