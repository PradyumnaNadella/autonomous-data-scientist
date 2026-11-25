# Train final model on full set and save
import pandas as pd
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

# Prepare data
train_df = pd.read_csv('train.csv')
train_df_f = build_features(train_df)
for col in ['CryoSleep', 'VIP']:
    train_df_f[col] = train_df_f[col].astype(str)
X = train_df_f.drop(columns=['Transported', 'PassengerId'], errors='ignore')
y = train_df_f['Transported'].astype(int)

# Rebuild preprocessor (needs to be consistent)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Final model pipeline
final_model = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, verbose=-1))])
final_model.fit(X, y)

# Predictions
test_df = pd.read_csv('test.csv')
test_df_f = build_features(test_df)
for col in ['CryoSleep', 'VIP']:
    test_df_f[col] = test_df_f[col].astype(str)
X_test = test_df_f.drop(columns=['PassengerId'], errors='ignore')

predictions = final_model.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': predictions.astype(bool)})
submission.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")
