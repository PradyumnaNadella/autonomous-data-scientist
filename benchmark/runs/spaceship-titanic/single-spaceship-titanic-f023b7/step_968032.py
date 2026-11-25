import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("\nMissing values (Train):\n", train.isnull().sum())
print("\nTarget distribution:\n", train['Transported'].value_counts(normalize=True))
print("\nInfo:\n", train.info())
