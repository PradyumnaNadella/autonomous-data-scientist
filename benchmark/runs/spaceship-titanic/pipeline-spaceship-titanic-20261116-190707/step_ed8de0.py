import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def summarize(df, name):
    print(f"--- {name} ---")
    print(f"Shape: {df.shape}")
    print(f"Missing:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Target Distribution:\n{df['Transported'].value_counts(normalize=True) if 'Transported' in df.columns else 'N/A'}")
    print(f"Columns: {list(df.columns)}")
    print("\n")

summarize(train, "Train")
summarize(test, "Test")

# Check for mixed information columns
print("Sample of 'PassengerId', 'Cabin', 'Name':")
print(train[['PassengerId', 'Cabin', 'Name']].head(3))
