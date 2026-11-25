# Train on full train set and predict test set
import pandas as pd
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

df_train = pd.read_csv('train.csv')
X_train = build_features(df_train.drop('Transported', axis=1))
y_train = df_train['Transported'].astype(int)
df_test = pd.read_csv('test.csv')
X_test = build_features(df_test)

# Fix types
for df in [X_train, X_test]:
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(str)
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)

cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
num_features = X_train.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_features),
    ('cat', Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_features)
])

final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42))
])

final_model.fit(X_train, y_train)
preds = final_model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Transported': preds.astype(bool)
})
submission.to_csv('predictions.csv', index=False)

# Get feature importances
importances = final_model.named_steps['model'].feature_importances_
feature_names = final_model.named_steps['preprocessor'].get_feature_names_out()
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("Top 5 features:")
print(feat_imp.head(5))
