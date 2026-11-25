from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
import pandas as pd

train = pd.read_csv('train.csv')
X = train.drop('Transported', axis=1)
y = train['Transported'].astype(int)

# Feature engineering
for df in [X]:
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)

num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_features)
    ])

clf_gb = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))])

cv_scores_gb = cross_val_score(clf_gb, X, y, cv=5, scoring='accuracy')
print(f"GB CV Score: {cv_scores_gb.mean():.4f}")
