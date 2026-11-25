import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load data
train = pd.read_csv('train.csv')

# Feature Engineering
def feature_engineering(df):
    df = df.copy()
    # Group from PassengerId
    df['Group'] = df['PassengerId'].str.split('_').str[0].astype(int)
    # Cabin: Deck, Side (ignore Num for simplicity)
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    # TotalSpend
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['IsSpending'] = (df['TotalSpend'] > 0).astype(int)
    # Drop Name, Cabin
    return df.drop(columns=['Name', 'Cabin', 'Num'])

train_proc = feature_engineering(train)
X = train_proc.drop(columns=['Transported', 'PassengerId'])
y = train_proc['Transported']

# Preprocessing
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'Group']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'IsSpending']

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

# Compare Models
models = [
    ('RF', RandomForestClassifier(random_state=42, n_estimators=100)),
    ('GB', GradientBoostingClassifier(random_state=42, n_estimators=100))
]

results = {}
for name, model in models:
    pipeline = Pipeline([('pre', preprocessor), ('model', model)])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name}: {scores.mean():.4f}")

# Train best model and predict
best_model = Pipeline([('pre', preprocessor), ('model', GradientBoostingClassifier(random_state=42, n_estimators=100))])
best_model.fit(X, y)

test = pd.read_csv('test.csv')
test_proc = feature_engineering(test)
X_test = test_proc.drop(columns=['PassengerId'])
predictions = best_model.predict(X_test)

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print("predictions.csv saved")
