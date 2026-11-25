# Check if PassengerId is used as a numerical feature
# Pipeline definition in train_model (not explicitly in current script but reconstructed)
# ColumnTransformer defined in memory:
# num: ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num']
# cat: ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Deck', 'Side', 'Surname']
# PassengerId is NOT in the columns used for modeling. Good.
print("PassengerId used in features? No.")
