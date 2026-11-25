import pandas as pd
from features import build_features
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Load data
train_df = pd.read_csv('train.csv')
X = build_features(train_df.drop('Transported', axis=1))
y = train_df['Transported']
X = X.drop(['PassengerId', 'Name'], axis=1, errors='ignore')

# Identify and convert
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'bool':
        X[col] = X[col].astype(str)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Simplified Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

# Model
model = RandomForestClassifier(n_estimators=50, random_state=42)
pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
score = cross_val_score(pipeline, X, y, cv=3, scoring='accuracy').mean()
print(f"RandomForest: {score:.4f}")
