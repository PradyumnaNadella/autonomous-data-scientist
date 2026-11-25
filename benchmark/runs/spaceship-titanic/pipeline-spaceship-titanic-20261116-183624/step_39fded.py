# Inspect column data types and unique values for potential complex strings
print(train.info())
print("\nUnique Cabin values (sample):", train['Cabin'].unique()[:5])
print("Unique PassengerId format:", train['PassengerId'].head(3).values)
