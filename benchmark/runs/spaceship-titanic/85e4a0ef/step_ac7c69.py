import pandas as pd
t = pd.read_csv('test.csv')
pd.DataFrame({'PassengerId': t.PassengerId, 'Transported': True}).to_csv('predictions.csv', index=False)