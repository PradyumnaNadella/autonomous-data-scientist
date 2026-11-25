import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv('train.csv')
X = train.drop(['row_id', 'tip'], axis=1)
y = train['tip']

cat_features = ['sex', 'smoker', 'day', 'time']
num_features = ['total_bill', 'size']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

gbr = Pipeline([('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor(random_state=42))])
cv_gbr = -cross_val_score(gbr, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
print(f"GBR MAE: {cv_gbr}")
