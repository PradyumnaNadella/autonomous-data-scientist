# Log experiments
import pandas as pd
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1. Log existing results
print("Log experiment LR: 0.7882")
print("Log experiment RF: 0.7862")

# Create train_model.py
train_script = """
import pandas as pd
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = build_features(train_df.drop('Transported', axis=1))
y_train = train_df['Transported'].astype(int)
X_test = build_features(test_df)

num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_cols = [c for c in X_train.columns if c not in num_cols and c != 'PassengerId']

for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

model = Pipeline([('pre', preprocessor), ('model', LogisticRegression(max_iter=1000))])
model.fit(X_train.drop('PassengerId', axis=1), y_train)

predictions = model.predict(X_test.drop('PassengerId', axis=1))
sub = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': predictions.astype(bool)})
sub.to_csv('predictions.csv', index=False)
"""
with open('train_model.py', 'w') as f:
    f.write(train_script)

# Run it
import subprocess
subprocess.run(['python', 'train_model.py'])
print("Predictions saved to predictions.csv")
