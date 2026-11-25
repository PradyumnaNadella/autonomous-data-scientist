from sklearn.ensemble import GradientBoostingRegressor

# GradientBoosting
gbr = Pipeline([('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor(random_state=42))])
cv_gbr = -cross_val_score(gbr, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
print(f"GBR MAE: {cv_gbr}")
