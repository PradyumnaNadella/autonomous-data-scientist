# Using just pandas first to verify read
try:
    df = pd.read_csv('train.csv')
    print(df.shape)
except Exception as e:
    print(e)
