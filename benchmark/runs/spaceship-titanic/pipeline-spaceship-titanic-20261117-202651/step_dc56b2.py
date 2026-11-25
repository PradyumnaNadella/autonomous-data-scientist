import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train columns: {train.columns}")
print(f"Test columns: {test.columns}")
# Check leakage in train_model.py
# The script reads train and test separately, builds features separately, 
# then does imputation within the pipeline. This is correct.
# BUT, the preprocessing in train_model.py is:
# 1. build_features (on full train)
# 2. build_features (on full test)
# 3. Then creates the pipeline using the FULL datasets' columns to define types.
# Wait, look at how categorical_features are derived in train_model.py:
# categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
# This looks okay because ColumnTransformer will handle the fit correctly.
# Let's verify if there is any leakage. 
# The issue is that train_model.py does:
# for col in X_train.columns: ... X_train[col] = X_train[col].astype(str) ... X_test[col] = ...
# This is fine. 
# The pipeline is: preprocessor -> classifier.
# Preprocessor uses SimpleImputer(strategy='median') on X_train.
# This is correctly fitted on X_train only during cross-validation 
# or model.fit(X_train, y_train).
# Looks like no leakage.
