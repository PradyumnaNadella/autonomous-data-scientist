# Check for leakage/differences
train_cols = set(train.columns)
test_cols = set(test.columns)
print(f"Columns in Train but not in Test: {train_cols - test_cols}")
print(f"Columns in Test but not in Train: {test_cols - train_cols}")

# Inspect Cabin and PassengerId structure
print("\nCabin unique structure examples:")
print(train['Cabin'].dropna().head(5).tolist())

print("\nPassengerId structure examples:")
print(train['PassengerId'].head(5).tolist())
