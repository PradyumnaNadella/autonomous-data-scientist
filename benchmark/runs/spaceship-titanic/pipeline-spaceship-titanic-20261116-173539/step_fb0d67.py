import pandas as pd
train = pd.read_csv('train.csv')
print(f"Train types: {train['Transported'].dtype}")
# The problem asked for same label format as train.csv.
# If it was True/False, they should be True/False, not 1/0.
