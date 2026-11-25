import pandas as pd
predictions = pd.read_csv('predictions.csv')
predictions['Transported'] = predictions['Transported'].astype(bool)
predictions.to_csv('predictions.csv', index=False)
print("Updated predictions format to boolean.")
