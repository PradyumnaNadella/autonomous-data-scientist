import pandas as pd

# Try simple approach without complex imports
try:
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("\nTrain info:")
    print(train.info())
    print("\nMissing values (Train):\n", train.isnull().sum())
    print("\nTarget:\n", train['Transported'].value_counts(normalize=True))
except Exception as e:
    print(e)
