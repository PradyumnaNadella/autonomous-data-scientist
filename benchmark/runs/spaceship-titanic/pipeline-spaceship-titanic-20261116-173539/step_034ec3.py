import subprocess

# Run the training script
subprocess.run(['python', 'train_model.py'], check=True)
print("Predictions saved to predictions.csv.")

# Print feature importances
import pandas as pd
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

train_df = build_features(pd.read_csv('train.csv'))
y = train_df['Transported'].astype(int)
X = train_df.drop(columns=['PassengerId', 'Name', 'Transported'], errors='ignore')

cat_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])

clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
model = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
model.fit(X, y)

# Get feature names
cat_names = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_features)
feature_names = num_features + list(cat_names)
importances = model.named_steps['classifier'].feature_importances_

importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
print(importances_df.head(5))
