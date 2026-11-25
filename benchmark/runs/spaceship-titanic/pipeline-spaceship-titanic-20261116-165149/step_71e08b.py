import pandas as pd
from features import build_features
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

# Prepare training data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Preprocessing
df_train = build_features(train.drop('Transported', axis=1))
df_train['GroupSize'] = df_train.groupby('Group')['Group'].transform('count')
y_train = train['Transported']

df_test = build_features(test)
df_test['GroupSize'] = df_test.groupby('Group')['Group'].transform('count')

# Features
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'GroupSize']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])

# Pipeline
model = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

# Train
pipeline.fit(df_train.drop(['PassengerId', 'Group', 'Member', 'Num', 'Surname'], axis=1), y_train)

# Predict
preds = pipeline.predict(df_test.drop(['PassengerId', 'Group', 'Member', 'Num', 'Surname'], axis=1))

# Save
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': preds})
submission.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")

# Get Feature Importances
importances = pipeline.named_steps['classifier'].feature_importances_
feature_names = (num_features + 
                 list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_features)))
fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
print(fi_df.sort_values(by='importance', ascending=False).head(5))
