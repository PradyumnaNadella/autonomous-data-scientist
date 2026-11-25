import pandas as pd
train_df = pd.read_csv('train.csv')
print("CryoSleep conversion:")
print(train_df['CryoSleep'].astype(str).unique())
