# Final training with XGBoost
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import pandas as pd
from features import build_features

train = build_features(pd.read_csv('train.csv'))
test = build_features(pd.read_csv('test.csv'))

X = train.drop(columns=['Transported'])
y = train['Transported']

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ]
)

final_model = Pipeline(steps=[('pre', preprocessor), ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
final_model.fit(X, y)

predictions = final_model.predict(test)
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Transported': predictions})
output.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
