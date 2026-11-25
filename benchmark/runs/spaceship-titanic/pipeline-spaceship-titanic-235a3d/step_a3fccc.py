from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('train.csv')
train_feat = build_features(train_df)
X = train_feat.drop(columns=['Transported', 'PassengerId'])
y = train_feat['Transported']

num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ]
)

pipeline = Pipeline([('preprocessor', preprocessor), ('model', LGBMClassifier(random_state=42, verbose=-1))])

param_grid = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print(f"Best score: {grid.best_score_:.4f}")
print(f"Best params: {grid.best_params_}")
