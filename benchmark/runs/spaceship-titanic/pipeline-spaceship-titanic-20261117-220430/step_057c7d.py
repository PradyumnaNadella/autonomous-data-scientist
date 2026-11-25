import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv('train.csv')
features = build_features(df)
X = features.drop(columns=['Transported', 'PassengerId'])
y = features['Transported'].astype(int)

# Inspect dtypes
print(X.dtypes)

# Convert boolean columns to strings for consistent encoding
bool_cols = X.select_dtypes(include=['bool']).columns
for col in bool_cols:
    X[col] = X[col].astype(str)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Models
rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
xgb = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])

# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_rf = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
scores_xgb = cross_val_score(xgb, X, y, cv=cv, scoring='accuracy')

print(f"RF CV: {scores_rf.mean():.4f}")
print(f"XGB CV: {scores_xgb.mean():.4f}")
