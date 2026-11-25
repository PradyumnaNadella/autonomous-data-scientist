import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
from features import build_features

# Load and process
train = pd.read_csv('train.csv')
train = build_features(train)
for col in train.select_dtypes(include=['object', 'bool']).columns:
    if col != 'Transported':
        train[col] = train[col].astype(str)

y = train['Transported']
X = train.drop('Transported', axis=1)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Tune LightGBM
pipeline = Pipeline([('preprocessor', preprocessor), ('model', LGBMClassifier(random_state=42, verbose=-1))])
param_grid = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1], 'model__max_depth': [5, 10]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

print(f"Best parameters: {grid.best_params_}")
print(f"Best CV Accuracy: {grid.best_score_:.4f}")

# Log tuned model
# I need to call log_experiment again
