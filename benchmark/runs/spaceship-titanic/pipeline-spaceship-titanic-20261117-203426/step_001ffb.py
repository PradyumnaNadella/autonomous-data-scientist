import pandas as pd
train = pd.read_csv('train.csv')
print(train['Surname'].value_counts() if 'Surname' in train.columns else 'Surname not in train')
from features import build_features
train_f = build_features(train)
print(f"Unique Surnames: {train_f['Surname'].nunique()}")
