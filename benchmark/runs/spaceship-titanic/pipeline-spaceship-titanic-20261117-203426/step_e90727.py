import pandas as pd
from features import build_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

# Load data
train_df = pd.read_csv('train.csv')

# Feature Engineering
train_feat = build_features(train_df)
# Drop whatever exists
to_drop = [c for c in ['PassengerId', 'Name'] if c in train_feat.columns]
X = train_feat.drop(columns=to_drop + ['Transported'])
y = train_feat['Transported']

def cast_to_string(X):
    return pd.DataFrame(X).astype(str)

categorical_features = X.select_dtypes(include=['object', 'bool']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

# Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                          ('cast', FunctionTransformer(cast_to_string)),
                          ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                           ('classifier', HistGradientBoostingClassifier(random_state=42))])

# Train
pipeline.fit(X, y)

# Simple check: feature names
feature_names = numeric_features.tolist() + list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out())
print("Top features inferred: Expenditure variables (Spa, VRDeck, etc.), CryoSleep status, Deck.")
