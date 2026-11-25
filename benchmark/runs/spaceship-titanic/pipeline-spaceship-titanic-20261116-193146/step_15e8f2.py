import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Cabin unique sample: {train['Cabin'].dropna().unique()[:5]}")
print(f"PassengerId format sample: {train['PassengerId'].head(3)}")
for col in ['HomePlanet', 'Destination', 'VIP', 'CryoSleep']:
    print(f"\n{col} Train unique: {train[col].unique()}")
    print(f"{col} Test unique: {test[col].unique()}")
