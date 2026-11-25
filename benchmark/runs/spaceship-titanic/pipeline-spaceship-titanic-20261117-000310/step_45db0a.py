# Final check on submission format
import pandas as pd
predictions = pd.read_csv('predictions.csv')
print(predictions['Transported'].dtype)
