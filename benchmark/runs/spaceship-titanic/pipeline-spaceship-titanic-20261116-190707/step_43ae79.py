# Convert Transported back to original format (bool) just in case
df = pd.read_csv('predictions.csv')
train = pd.read_csv('train.csv')
df['Transported'] = df['Transported'].astype(bool)
df.to_csv('predictions.csv', index=False)
print(df.dtypes)
