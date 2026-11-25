import pandas as pd
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Feature Engineering
train_df = build_features(train_df)
test_df = build_features(test_df)

# Drop PassengerId and Name (as per plan)
train_df = train_df.drop(columns=['PassengerId', 'Name'], errors='ignore')
test_df = test_df.drop(columns=['PassengerId', 'Name'], errors='ignore')

# Identify feature types
cat_features = train_df.select_dtypes(include=['object', 'bool']).columns.tolist()
num_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Transported' in num_features:
    num_features.remove('Transported')

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])

# Model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss'))
])

# Save as train_model.py
script = """
import pandas as pd
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submission_ids = test_df['PassengerId']

# Feature Engineering
train_df = build_features(train_df)
test_df = build_features(test_df)

# Drop PassengerId and Name
train_df = train_df.drop(columns=['PassengerId', 'Name'], errors='ignore')
test_df = test_df.drop(columns=['PassengerId', 'Name'], errors='ignore')

# Target
y = train_df['Transported'].astype(int)
X = train_df.drop(columns=['Transported'])

# Identify feature types
cat_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])

# Model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss'))
])

# Train
model.fit(X, y)

# Predict
preds = model.predict(test_df)
submission = pd.DataFrame({'PassengerId': submission_ids, 'Transported': preds.astype(bool)})
submission.to_csv('predictions.csv', index=False)
"""

with open('train_model.py', 'w') as f:
    f.write(script)

print("train_model.py created.")
