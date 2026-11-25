# Attempt to use a different approach if write is blocked. 
# Actually, the error might be because I don't have permission to write in the current directory?
# Let's try reading the files instead to verify I can read.
import pandas as pd
train = pd.read_csv('train.csv')
print(train.columns)
