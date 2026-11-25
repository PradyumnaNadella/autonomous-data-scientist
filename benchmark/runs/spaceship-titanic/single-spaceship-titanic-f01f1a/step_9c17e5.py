import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("-" * 20)
print(train.info())
print("-" * 20)
print(f"Missing values:\n{train.isnull().sum()}")
print("-" * 20)
print(f"Target distribution:\n{train['Transported'].value_counts(normalize=True)}")
