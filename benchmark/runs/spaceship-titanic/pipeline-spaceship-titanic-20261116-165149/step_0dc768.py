# Verify if the train target is indeed boolean
train = pd.read_csv('train.csv')
print(f"Train Transported type: {train['Transported'].dtype}")
print(f"Train unique values: {train['Transported'].unique()}")
