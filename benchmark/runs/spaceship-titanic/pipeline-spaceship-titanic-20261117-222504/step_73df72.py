import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
from features import build_features
from sklearn.model_selection import GridSearchCV

# Re-run the tuning to have the model available
train_df = pd.read_csv('train.csv')
X = build_features(train_df.drop(columns=['Transported']))
y = train_df['Transported'].astype(int)
for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'GroupId', 'Deck', 'Side']:
    X[col] = X[col].astype(str)

num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum', 'TotalSpent', 'HasSpent']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'GroupId', 'Deck', 'Side']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), 
                      ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

pipe = Pipeline([('preprocessor', preprocessor), ('model', LGBMClassifier(random_state=42, verbose=-1, learning_rate=0.05, max_depth=5, n_estimators=200))])
pipe.fit(X, y)

# Get feature importance
model = pipe.named_steps['model']
preprocessor = pipe.named_steps['preprocessor']
num_cols = num_features
cat_cols = preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_features)
all_features = np.concatenate([num_cols, cat_cols])

importances = model.feature_importances_
feat_importances = pd.Series(importances, index=all_features)
print(feat_importances.nlargest(5))

# Save the model
import joblib
joblib.dump(pipe, 'model.joblib')
