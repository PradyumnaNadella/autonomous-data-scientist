import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
from features import build_features

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Preprocess
X = build_features(train).drop(columns=['Transported'])
y = train['Transported']
X_test = build_features(test)

# Pipeline
cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(learning_rate=0.05, n_estimators=100, num_leaves=31, verbose=-1))
])

# Fit on full data
model.fit(X, y)

# Predict
predictions = model.predict(X_test)

# Save
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
output.to_csv('predictions.csv', index=False)
print("predictions.csv saved.")
