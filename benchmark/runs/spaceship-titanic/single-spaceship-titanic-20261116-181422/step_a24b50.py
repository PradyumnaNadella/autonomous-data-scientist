import pandas as pd
submission = pd.read_csv('predictions.csv')
# The target was originally boolean, but the predictions are 1/0. 
# Let's check the submission format. 
# The instructions say 'same label format as train.csv'
# Train.csv target is bool.
# Let's ensure the output is boolean.
submission['Transported'] = submission['Transported'].astype(bool)
submission.to_csv('predictions.csv', index=False)
print(submission.head())
print(submission.dtypes)
