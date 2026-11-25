import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

# Load data
train_df = pd.read_csv('train.csv')

def process_data(df):
    df = df.copy()
    # Explicitly convert object/bool to string for safety
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            df[col] = df[col].astype(str)
    
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Surname'] = df['Name'].apply(lambda x: x.split()[-1] if pd.notnull(x) else 'Unknown')
    return df.drop(columns=['PassengerId', 'Name', 'Cabin'])

X = process_data(train_df.drop(columns=['Transported']))
y = train_df['Transported']

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# Need to handle Num (the split part)
X['Num'] = pd.to_numeric(X['Num'], errors='coerce')
numeric_features.append('Num')

categorical_features = [c for c in X.columns if c not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

# CV
model = HistGradientBoostingClassifier(random_state=42)
pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(n_splits=3), scoring='accuracy')
print(f"HGB CV Accuracy: {scores.mean():.4f}")
