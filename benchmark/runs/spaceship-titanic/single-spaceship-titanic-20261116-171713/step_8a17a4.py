import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain info:")
print(train.info())
print("\nTrain missing values:")
print(train.isnull().sum())
print("\nTarget distribution:")
print(train['Transported'].value_counts(normalize=True))
