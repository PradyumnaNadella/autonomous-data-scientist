import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature Engineering
def feature_engineering(df):
    df = df.copy()
    # PassengerId: Group
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    # Cabin: Deck, Num, Side
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    # TotalSpend
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['IsSpending'] = (df['TotalSpend'] > 0).astype(int)
    # Drop columns that are too high cardinality for simple OneHot
    drop_cols = ['PassengerId', 'Name', 'Cabin', 'Num']
    return df.drop(columns=drop_cols)

train_proc = feature_engineering(train)
X = train_proc.drop(columns=['Transported'])
y = train_proc['Transported']

# Preprocessing
num_cols = X.select_dtypes(include=['number']).columns
cat_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

# Compare Models
models = [
    ('RF', RandomForestClassifier(random_state=42)),
    ('GB', GradientBoostingClassifier(random_state=42))
]

for name, model in models:
    pipeline = Pipeline([('pre', preprocessor), ('model', model)])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")
