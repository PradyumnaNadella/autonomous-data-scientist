# Investigate Cabin and PassengerId
print(f"Cabin unique sample: {train['Cabin'].dropna().unique()[:5]}")
print(f"PassengerId format sample: {train['PassengerId'].head(3)}")
# Check for possible leakage/distribution drift by checking categorical column unique values
for col in ['HomePlanet', 'Destination', 'VIP', 'CryoSleep']:
    print(f"\n{col} Train unique: {train[col].unique()}")
    print(f"{col} Test unique: {test[col].unique()}")
