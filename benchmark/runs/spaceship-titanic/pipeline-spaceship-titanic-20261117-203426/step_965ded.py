import pandas as pd
import json
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load data
train_df = pd.read_csv('train.csv')

# Feature processing
def process_data(df):
    df = df.copy()
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Surname'] = df['Name'].apply(lambda x: x.split()[-1] if pd.notnull(x) else 'Unknown')
    return df.drop(columns=['PassengerId', 'Name', 'Cabin'])

X = process_data(train_df.drop(columns=['Transported']))
y = train_df['Transported']

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']
categorical_features = [c for c in X.columns if c not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

# Models
models = {
    'RF': RandomForestClassifier(n_estimators=50, random_state=42),
    'HGB': HistGradientBoostingClassifier(random_state=42)
}

# CV
results = {}
for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(n_splits=3), scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name}: {scores.mean():.4f}")

print(f"Results: {results}")
