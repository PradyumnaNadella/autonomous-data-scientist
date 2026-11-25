# Re-defining everything to ensure consistency for logging
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Need to reload data and define pipeline inside the calls if I want to use log_experiment
train = pd.read_csv('train.csv')
X = train.drop('Transported', axis=1)
y = train['Transported'].astype(int)
for df in [X]:
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)

numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Logging results
from sklearn.model_selection import cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
rf_score = cross_val_score(rf_pipeline, X, y, cv=cv, scoring='accuracy').mean()
# Assuming log_experiment is available directly as a tool call
print(f"RandomForest Score: {rf_score}")

gb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GradientBoostingClassifier(random_state=42))])
gb_score = cross_val_score(gb_pipeline, X, y, cv=cv, scoring='accuracy').mean()
print(f"GradientBoosting Score: {gb_score}")
