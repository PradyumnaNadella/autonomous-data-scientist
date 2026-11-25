# Check feature importance of the best model
best_model = grid.best_estimator_
model = best_model.named_steps['model']
preprocessor = best_model.named_steps['preprocessor']

# Get feature names
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum', 'TotalSpent', 'HasSpent']
cat_cols = preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'GroupId', 'Deck', 'Side'])
all_features = np.concatenate([num_cols, cat_cols])

importances = model.feature_importances_
feat_importances = pd.Series(importances, index=all_features)
print(feat_importances.nlargest(5))
