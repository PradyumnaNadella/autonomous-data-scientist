# Prepare feature importance
model_lgbm_tuned = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1))])
model_lgbm_tuned.fit(X_train, y_train)

# Feature importance
importances = model_lgbm_tuned.named_steps['classifier'].feature_importances_
feature_names = numeric_features + list(model_lgbm_tuned.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(categorical_features))

feat_imp = pd.Series(importances, index=feature_names)
print(feat_imp.sort_values(ascending=False).head(5))
