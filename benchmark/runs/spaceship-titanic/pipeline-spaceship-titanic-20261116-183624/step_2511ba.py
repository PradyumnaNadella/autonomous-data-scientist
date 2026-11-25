import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from features import build_features

# Load and prepare
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
X = train_df.drop('Transported', axis=1)
y = train_df['Transported'].astype(int)

# Feature engineering
X_eng = build_features(X)
X_test_eng = build_features(test_df)

# Cast types for consistency
for col in X_eng.columns:
    if X_eng[col].dtype == 'object' or X_eng[col].dtype == 'bool':
        X_eng[col] = X_eng[col].astype(str)
        X_test_eng[col] = X_test_eng[col].astype(str)

drop_cols = ['PassengerId', 'Name']
X_eng = X_eng.drop(columns=[c for c in drop_cols if c in X_eng.columns])
X_test_eng = X_test_eng.drop(columns=[c for c in drop_cols if c in X_test_eng.columns])

numeric_features = X_eng.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_eng.select_dtypes(include=['object']).columns

# Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

model = Pipeline([('preprocessor', preprocessor), 
                  ('classifier', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss'))])

model.fit(X_eng, y)

# Predict
preds = model.predict(X_test_eng)
# Map back to boolean
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': preds.astype(bool)})
submission.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
