import pandas as pd
train = pd.read_csv('train.csv')
print(train.groupby('Transported')['RoomService'].mean())
print(train.groupby('Transported')['FoodCourt'].mean())
print(train.groupby('Transported')['VRDeck'].mean())
print(train.groupby('Transported')['Spa'].mean())
print(train.groupby('Transported')['TotalSpent'].mean() if 'TotalSpent' in train.columns else 'No TotalSpent yet')
