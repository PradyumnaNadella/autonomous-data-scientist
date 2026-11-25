import pandas as pd
train_df = pd.read_csv('train.csv')
print("CryoSleep unique values:")
print(train_df['CryoSleep'].unique())
print("CryoSleep type:")
print(train_df['CryoSleep'].dtype)
