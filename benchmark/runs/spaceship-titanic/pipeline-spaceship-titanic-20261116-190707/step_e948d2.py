# Create train_model.py
train_model_script = """
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from features import build_features

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preprocess
X_train = build_features(train_df.drop('Transported', axis=1))
y_train = train_df['Transported'].astype(int)
X_test = build_features(test_df)

# Type handling and drop
drop_cols = ['PassengerId', 'Surname', 'Group', 'Num']
for df in [X_train, X_test]:
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            df[col] = df[col].astype(str).replace('nan', 'missing')
    df.drop(columns=drop_cols, inplace=True)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Model
model = Pipeline(steps=[('preprocessor', preprocessor), 
                        ('classifier', XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric='logloss'))])
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': predictions.astype(bool)})
submission.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
"""

with open('train_model.py', 'w') as f:
    f.write(train_model_script)
print("train_model.py created")
