import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from features import build_features

# 1. Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. Build features
train_feat = build_features(train)
test_feat = build_features(test)

cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Num', 'Side', 'Surname']
for col in cat_cols:
    train_feat[col] = train_feat[col].astype(str)
    test_feat[col] = test_feat[col].astype(str)

X_train = train_feat.drop(columns=['Transported', 'PassengerId'])
y_train = train_feat['Transported'].astype(int)
X_test = test_feat.drop(columns=['PassengerId'])

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = [col for col in X_train.columns if col not in numeric_features]

# 3. Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Best parameters: {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
model = Pipeline(steps=[('preprocessor', preprocessor), 
                        ('classifier', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss'))])

# 4. Train
model.fit(X_train, y_train)

# 5. Predict
predictions = model.predict(X_test).astype(bool)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
submission.to_csv('predictions.csv', index=False)

# 6. Feature Importance (Rough estimate)
feature_names = (numeric_features + 
                 list(model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out()))
importances = model.named_steps['classifier'].feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("Top 5 Feature Importances:")
print(feat_imp.head(5))
