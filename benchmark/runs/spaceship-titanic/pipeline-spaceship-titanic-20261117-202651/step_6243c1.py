import pandas as pd
from features import build_features
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Prepare data
train_df = pd.read_csv('train.csv')
train_df_f = build_features(train_df)
for col in ['CryoSleep', 'VIP']:
    train_df_f[col] = train_df_f[col].astype(str)
X = train_df_f.drop(columns=['Transported', 'PassengerId'], errors='ignore')
y = train_df_f['Transported'].astype(int)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

lgbm = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', LGBMClassifier(random_state=42, verbose=-1))])

# Tuning
param_distributions = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.05, 0.1, 0.2],
    'classifier__max_depth': [5, 10, 15]
}

lgbm_tuned = RandomizedSearchCV(
    lgbm, param_distributions, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1
)
lgbm_tuned.fit(X, y)
print(f"Best parameters: {lgbm_tuned.best_params_}")
print(f"Tuned LGBM CV Accuracy: {lgbm_tuned.best_score_:.4f}")
