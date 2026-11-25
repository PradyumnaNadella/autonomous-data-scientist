import pandas as pd
from features import build_features
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load
train_df = pd.read_csv('train.csv')
train = build_features(train_df.copy())
target = 'Transported'

# Prepare data
if 'Name' in train.columns:
    train = train.drop(columns=['Name'])
train = train.drop(columns=['PassengerId'])

numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in train.columns:
    if col != target and col not in numeric_cols:
        train[col] = train[col].astype(str)

X = train.drop(columns=[target])
y = train[target].astype(int)
categorical_features = [c for c in X.columns if c not in numeric_cols]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_features)])

# Random Forest Comparison
rf_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42, n_estimators=50))])
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
rf_scores = cross_val_score(rf_clf, X, y, cv=cv, scoring='accuracy')
print(f"RandomForest: {rf_scores.mean():.4f}")

# HistGB is already 0.7991
# Save model script
with open('train_model.py', 'w') as f:
    f.write("""
import pandas as pd
from features import build_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
target = 'Transported'

def prep(df):
    if 'Name' in df.columns: df = df.drop(columns=['Name'])
    pid = df['PassengerId']
    df = df.drop(columns=['PassengerId'])
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in df.columns:
        if col != target and col not in numeric_cols:
            df[col] = df[col].astype(str)
    return df, pid

train, _ = prep(train_df)
test, test_pid = prep(test_df)
X = train.drop(columns=[target])
y = train[target].astype(int)
numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = [c for c in X.columns if c not in numeric_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)])

clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', HistGradientBoostingClassifier(random_state=42))])
clf.fit(X, y)
preds = clf.predict(test)
pd.DataFrame({'PassengerId': test_pid, 'Transported': preds.astype(bool)}).to_csv('predictions.csv', index=False)
""")
