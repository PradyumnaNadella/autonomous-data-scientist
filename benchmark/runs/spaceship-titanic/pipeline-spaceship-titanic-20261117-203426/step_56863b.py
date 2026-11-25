import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Re-train on full data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

def process_data(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            df[col] = df[col].astype(str)
    
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Surname'] = df['Name'].apply(lambda x: x.split()[-1] if pd.notnull(x) else 'Unknown')
    return df.drop(columns=['PassengerId', 'Name', 'Cabin'])

X = process_data(train_df.drop(columns=['Transported']))
y = train_df['Transported']
X_test = process_data(test_df)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
X['Num'] = pd.to_numeric(X['Num'], errors='coerce')
X_test['Num'] = pd.to_numeric(X_test['Num'], errors='coerce')
numeric_features.append('Num')
categorical_features = [c for c in X.columns if c not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

pipe = Pipeline([('preprocessor', preprocessor), ('classifier', HistGradientBoostingClassifier(random_state=42))])
pipe.fit(X, y)
preds = pipe.predict(X_test)
pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': preds}).to_csv('predictions.csv', index=False)
print("Predictions saved.")
