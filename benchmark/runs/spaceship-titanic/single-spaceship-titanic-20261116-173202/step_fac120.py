import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Preprocessing function
def preprocess_features(df):
    df = df.copy()
    df['CryoSleep'] = df['CryoSleep'].astype(str)
    df['VIP'] = df['VIP'].astype(str)
    df['Cabin'] = df['Cabin'].fillna('Unknown/Unknown/Unknown')
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    return df.drop(['PassengerId', 'Name', 'Cabin'], axis=1)

X = preprocess_features(train.drop('Transported', axis=1))
y = train['Transported']
X_test = preprocess_features(test)

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

# Final model
model = Pipeline([('preprocessor', preprocessor), ('clf', GradientBoostingClassifier(random_state=42))])
model.fit(X, y)

# Predict
predictions = model.predict(X_test)

# Save
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
