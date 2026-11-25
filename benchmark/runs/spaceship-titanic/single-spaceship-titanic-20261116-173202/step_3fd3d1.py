import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')

def preprocess_features(df):
    df = df.copy()
    # Handle boolean/mixed types explicitly
    df['CryoSleep'] = df['CryoSleep'].astype(str)
    df['VIP'] = df['VIP'].astype(str)
    
    # Cabin
    df['Cabin'] = df['Cabin'].fillna('Unknown/Unknown/Unknown')
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    
    return df.drop(['PassengerId', 'Name', 'Cabin'], axis=1)

train_processed = preprocess_features(train)
X = train_processed.drop('Transported', axis=1)
y = train_processed['Transported'].astype(int)

# Identify features - ensure everything is treated as expected
cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

# Pipeline
model_rf = Pipeline([('preprocessor', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
model_gb = Pipeline([('preprocessor', preprocessor), ('clf', GradientBoostingClassifier(random_state=42))])

# CV
scores_rf = cross_val_score(model_rf, X, y, cv=5, scoring='accuracy')
scores_gb = cross_val_score(model_gb, X, y, cv=5, scoring='accuracy')

print(f"RF CV Score: {scores_rf.mean():.4f}")
print(f"GB CV Score: {scores_gb.mean():.4f}")
