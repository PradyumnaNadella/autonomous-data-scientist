import pandas as pd
import numpy as np

def load_data():
    return pd.read_csv('train.csv'), pd.read_csv('test.csv')

train, test = load_data()
print("Successfully read data using standard pandas!")
print(train.head(1))
