import pandas as pd
from features import build_features
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Build features
train_df = build_features(train)
test_df = build_features(test)

# Map target
target = 'Transported'
le = LabelEncoder()
train_df[target] = le.fit_transform(train_df[target])

# Identify features
features = [col for col in train_df.columns if col not in ['Transported', 'PassengerId', 'Name']]

# Ensure types are uniform for OHE
numeric_features = train_df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = train_df[features].select_dtypes(include=['object', 'bool']).columns.tolist()

# Ensure object columns are string
for col in categorical_features:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))])

# Tuning
param_dist = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [3, 5],
}

random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=4, cv=5, scoring='accuracy', random_state=42)
random_search.fit(train_df[features], train_df[target])

print(f"Best params: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
