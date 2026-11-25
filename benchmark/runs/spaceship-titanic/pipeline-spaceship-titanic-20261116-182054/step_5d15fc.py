# Check leakage in the original training script
# The train_model.py provided:
# preprocessor = ColumnTransformer(transformers=[
#     ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
#     ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
# ])
# model = Pipeline([('pre', preprocessor), ('model', LogisticRegression(max_iter=1000))])

# It seems correct, the issue is likely the 'bool' to 'str' conversion in the training data preparation.
# The user's code failed because `train_df[col] = train_df[col].astype(str)` was not done before fitting in the *user* provided `train_model.py`.

# I need to approve or reject. The user's provided `train_model.py` had a logic error.
# The user's reported CV scores (0.7882 for LR) actually match my computation.
# Wait, if they match, maybe they had the logic in `features.py`? Let me re-check `features.py`.
# No, `features.py` does not have conversion.
