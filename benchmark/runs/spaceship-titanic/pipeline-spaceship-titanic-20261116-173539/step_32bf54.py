# Get feature importances from the trained model
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

# Re-load data for feature importance
def build_features(df):
    df = df.copy()
    df['Group'] = df['PassengerId'].str.split('_').str[0].astype(int)
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Num'] = pd.to_numeric(cabin_split[1])
    df['Side'] = cabin_split[2]
    df = df.drop(columns=['Name', 'Cabin'])
    return df

train_df = build_features(pd.read_csv('train.csv'))
target = 'Transported'
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num', 'Group']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side', 'VIP']
for col in cat_features:
    train_df[col] = train_df[col].astype(str)

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])
clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
pipeline = Pipeline([('prep', preprocessor), ('clf', clf)])
pipeline.fit(train_df.drop(columns=[target, 'PassengerId']), train_df[target])

importances = clf.feature_importances_
feature_names = num_features + list(pipeline.named_steps['prep'].named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_features))
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feat_imp.head(5))
