# Create train_model.py
train_script = """
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

def build_features(df):
    df = df.copy()
    df['Group'] = df['PassengerId'].str.split('_').str[0].astype(int)
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = pd.to_numeric(cabin_split[1])
    df['Side'] = cabin_split[2]
    df = df.drop(columns=['Name', 'Cabin'])
    return df

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df = build_features(train_df)
test_df = build_features(test_df)

target = 'Transported'
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num', 'Group']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'VIP']

for col in cat_features:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

# Use best params found
clf = Pipeline([
    ('prep', preprocessor),
    ('clf', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42))
])

clf.fit(train_df.drop(columns=[target, 'PassengerId']), train_df[target])
preds = clf.predict(test_df.drop(columns=['PassengerId']))

output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': preds})
output.to_csv('predictions.csv', index=False)
"""

with open('train_model.py', 'w') as f:
    f.write(train_script)
