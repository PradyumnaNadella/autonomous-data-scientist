from features import build_features
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data
df = pd.read_csv('train.csv')
X = df.drop('Transported', axis=1)
y = df['Transported'].astype(int)
X = build_features(X)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'bool']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

xgb = Pipeline([('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42))])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.05, 0.1]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Best Params: {grid_search.best_params_}")

# Feature importance
best_model = grid_search.best_estimator_
cat_ohe = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
cat_names = cat_ohe.get_feature_names_out(categorical_features)
all_names = list(numeric_features) + list(cat_names)
importances = best_model.named_steps['classifier'].feature_importances_
feature_imp = pd.Series(importances, index=all_names).sort_values(ascending=False)
print("Top 5 Features:")
print(feature_imp.head(5))
