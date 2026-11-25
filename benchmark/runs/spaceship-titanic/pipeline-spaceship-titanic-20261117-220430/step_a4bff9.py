import pandas as pd
from features import build_features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# Re-setup
train_df = pd.read_csv('train.csv')
train_df = build_features(train_df)
X = train_df.drop(columns=['Transported', 'PassengerId'])
y = train_df['Transported'].astype(int)

cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Group']
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_cols)
])

# Re-fit the best model manually to extract importances
best_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss',
                            n_estimators=100, max_depth=5, learning_rate=0.05))
])
best_model.fit(X, y)

cat_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
all_names = num_cols + list(cat_names)
importances = best_model.named_steps['model'].feature_importances_
feature_imp = pd.Series(importances, index=all_names).sort_values(ascending=False)
print(feature_imp.head(5))
