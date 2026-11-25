import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

train_df = pd.read_csv('train.csv')
X = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported'].astype(int)

# Need to ensure types are handled consistently
for col in X.select_dtypes(include=['object', 'bool']).columns:
    X[col] = X[col].astype(str)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

# CV evaluation
pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100, use_label_encoder=False, eval_metric='logloss'))
])

scores = cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
print(f"Mean CV Accuracy: {scores.mean():.4f}")
