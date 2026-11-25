import pandas as pd
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Build features
train_df = build_features(train)
test_df = build_features(test)

target = 'Transported'
le = LabelEncoder()
train_df[target] = le.fit_transform(train_df[target])

features = [col for col in train_df.columns if col not in ['Transported', 'PassengerId', 'Name']]

numeric_features = train_df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = train_df[features].select_dtypes(include=['object', 'bool']).columns.tolist()

for col in categorical_features:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Chosen best params
model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='logloss')
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

# Fit
final_pipeline.fit(train_df[features], train_df[target])

# Feature importance
feature_names = numeric_features + list(final_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features))
importances = model.feature_importances_
feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
print(feature_importance.sort_values(by='importance', ascending=False).head(5))

# Predictions
preds = final_pipeline.predict(test_df[features])
test_df['Transported'] = le.inverse_transform(preds)
test_df[['PassengerId', 'Transported']].to_csv('predictions.csv', index=False)
