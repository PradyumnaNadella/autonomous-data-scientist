import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Just perform the logic in memory, avoiding complex dependencies if possible
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def process(df):
    df = df.copy()
    c = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = c[0].fillna('M')
    df['Side'] = c[2].fillna('S')
    billing = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[billing] = df[billing].fillna(0)
    df['TotalSpend'] = df[billing].sum(axis=1)
    return df.drop(columns=['PassengerId', 'Name', 'Cabin'])

X = process(train.drop(columns=['Transported']))
y = train['Transported'].astype(int)
X_test = process(test)

num = X.select_dtypes(include=['number']).columns.tolist()
cat = X.select_dtypes(include=['object']).columns.tolist()

pre = ColumnTransformer([
    ('n', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), num),
    ('c', Pipeline([('i', SimpleImputer(strategy='most_frequent')), ('o', OneHotEncoder(handle_unknown='ignore'))]), cat)
])

# Evaluation
model_rf = Pipeline([('p', pre), ('c', RandomForestClassifier(n_estimators=100, random_state=42))])
model_xgb = Pipeline([('p', pre), ('c', XGBClassifier(n_estimators=100, random_state=42))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("RF CV:", cross_val_score(model_rf, X, y, cv=cv).mean())
print("XGB CV:", cross_val_score(model_xgb, X, y, cv=cv).mean())

# Tune XGB
model_xgb_tuned = Pipeline([('p', pre), ('c', XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42))])
print("XGB Tuned CV:", cross_val_score(model_xgb_tuned, X, y, cv=cv).mean())

# Finalize
model_xgb_tuned.fit(X, y)
preds = model_xgb_tuned.predict(X_test)
out = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds.astype(bool)})
out.to_csv('predictions.csv', index=False)
