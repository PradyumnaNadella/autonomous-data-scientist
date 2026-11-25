import pandas as pd
from features import build_features
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

train_df = pd.read_csv('train.csv')
train_df_f = build_features(train_df)

# Fix types: CryoSleep, VIP to string for now to avoid the encoder error
for col in ['CryoSleep', 'VIP']:
    train_df_f[col] = train_df_f[col].astype(str)

X = train_df_f.drop(columns=['Transported', 'PassengerId'], errors='ignore')
y = train_df_f['Transported'].astype(int)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42))])

lgbm = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', LGBMClassifier(random_state=42, verbose=-1))])

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in [('RandomForest', rf), ('LGBM', lgbm)]:
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"{name} CV Accuracy: {scores.mean():.4f}")
