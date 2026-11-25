import pandas as pd
train_df = pd.read_csv('train.csv')
print(train_df['CryoSleep'].unique())
print(train_df['VIP'].unique())
