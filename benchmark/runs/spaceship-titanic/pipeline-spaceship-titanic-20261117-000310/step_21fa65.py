import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

train_df = pd.read_csv('train.csv')
y = train_df['Transported'].astype(int)
X = build_features(train_df.drop('Transported', axis=1)).drop('PassengerId', axis=1)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Re-run Cross Validation to verify accuracy
model = Pipeline(steps=[('preprocessor', preprocessor), 
                        ('classifier', LGBMClassifier(learning_rate=0.05, max_depth=10, n_estimators=100, random_state=42, verbose=-1))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f}")
