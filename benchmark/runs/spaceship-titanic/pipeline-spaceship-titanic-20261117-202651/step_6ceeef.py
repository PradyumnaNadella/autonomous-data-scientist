import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMClassifier
from features import build_features

# Load data
train = pd.read_csv('train.csv')
X_train = build_features(train.drop(columns=['Transported']))
y_train = train['Transported']

# Convert object/bool columns to string to ensure OHE works
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].astype(str)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Pipeline
model_lgbm = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LGBMClassifier(random_state=42, verbose=-1))])

# Tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, 15],
    'classifier__learning_rate': [0.05, 0.1]
}

grid_search = GridSearchCV(model_lgbm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best CV: {grid_search.best_score_}")
print(f"Best Params: {grid_search.best_params_}")
