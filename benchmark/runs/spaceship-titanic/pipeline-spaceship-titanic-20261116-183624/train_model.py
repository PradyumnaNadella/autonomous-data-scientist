
import pandas as pd
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

def train_and_predict():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    X = build_features(train_df.drop('Transported', axis=1))
    y = train_df['Transported'].astype(int)
    X_test = build_features(test_df)

    for col in X.select_dtypes(include=['object', 'bool']).columns:
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', Pipeline([
                ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
                ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

    model = Pipeline([
        ('prep', preprocessor),
        ('clf', XGBClassifier(learning_rate=0.05, max_depth=5, n_estimators=100, use_label_encoder=False, eval_metric='logloss'))
    ])

    model.fit(X, y)
    predictions = model.predict(X_test)
    
    output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': predictions.astype(bool)})
    output.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == '__main__':
    train_and_predict()
