# Analyze 'Cabin' and 'PassengerId' structures
print("PassengerId example:", train['PassengerId'].iloc[0])
print("Cabin example:", train['Cabin'].dropna().iloc[0])
print("\nUnique count in PassengerId:", train['PassengerId'].nunique())
print("Unique count in Cabin:", train['Cabin'].nunique())
