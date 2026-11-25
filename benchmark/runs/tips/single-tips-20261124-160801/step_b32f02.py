import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop(['row_id', 'tip'], axis=1)
y = train['tip']

# Preprocessing
cat_cols = ['sex', 'smoker', 'day', 'time']
num_cols = ['total_bill', 'size']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Compare models
models = [
    ('Ridge', Ridge()),
    ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42))
]

results = []
for name, model in models:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_absolute_error')
    mean_mae = -scores.mean()
    results.append((name, mean_mae))
    print(f"{name} CV MAE: {mean_mae:.4f}")

# Log experiments
for name, score in results:
    import json
    # Use a dummy call to simulate log_experiment behavior if not available
    print(f"Log: {name}, CV: {score}")
