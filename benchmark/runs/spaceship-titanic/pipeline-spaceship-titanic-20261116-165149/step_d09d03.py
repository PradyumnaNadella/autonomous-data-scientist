# Save the script as train_model.py
script = """
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from features import build_features

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    X = build_features(train).drop(columns=['Transported'])
    y = train['Transported']
    X_test = build_features(test)

    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(exclude=['object']).columns

    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(learning_rate=0.05, n_estimators=100, num_leaves=31, verbose=-1))
    ])

    model.fit(X, y)
    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
    output.to_csv('predictions.csv', index=False)
    print("predictions.csv created successfully.")

if __name__ == '__main__':
    main()
"""

with open('train_model.py', 'w') as f:
    f.write(script)
print("train_model.py saved.")
