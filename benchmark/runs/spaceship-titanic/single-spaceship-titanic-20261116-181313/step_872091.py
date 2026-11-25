from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Re-prepare for final prediction
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def extract_cabin(df):
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df = df.drop(columns=['Cabin', 'Name', 'Num'])
    return df

train = extract_cabin(train)
test = extract_cabin(test)

X = train.drop(columns=['Transported'])
y = train['Transported']

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ]
)

# Use LGBM as it performed better
final_pipe = Pipeline([('pre', preprocessor), ('clf', LGBMClassifier(random_state=42, verbose=-1))])
final_pipe.fit(X, y)

preds = final_pipe.predict(test)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds})
submission.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
