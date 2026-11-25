import pandas as pd
# Check if predictions are submitted as per train.csv format
# The task says: "using the same label format as train.csv"
# train['Transported'] is boolean.
# My check in train_model.py: 
# sub = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': predictions.astype(bool)})
# The model.predict returns the labels in the same format as y_train (which was booleans).
# So predictions are booleans.

# Wait, let me check the predictions.csv content I got earlier
sub = pd.read_csv('predictions.csv')
print(sub.head())
# sub['Transported'] is True/False.

# Looks good.
