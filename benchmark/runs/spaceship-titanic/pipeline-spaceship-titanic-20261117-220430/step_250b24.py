# Get feature names after onehot encoding
# The pipeline does not easily reveal names after fit, manual check
preprocessor.fit(X)
cat_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
all_names = num_cols + list(cat_names)
importances = search.best_estimator_.named_steps['model'].feature_importances_
feature_imp = pd.Series(importances, index=all_names).sort_values(ascending=False)
print(feature_imp.head(5))
