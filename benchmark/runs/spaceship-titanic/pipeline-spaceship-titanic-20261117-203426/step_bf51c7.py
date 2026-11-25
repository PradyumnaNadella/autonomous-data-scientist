import pandas as pd
train = pd.read_csv('train.csv')
print("Train shape:", train.shape)
print("Numeric columns missing/imputed:", train[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].isnull().sum())
# Verify if train_model.py actually used the features.py as claimed in reports/logs
with open('train_model.py', 'r') as f:
    code = f.read()
    print("Uses features.py:", "build_features" in code)
