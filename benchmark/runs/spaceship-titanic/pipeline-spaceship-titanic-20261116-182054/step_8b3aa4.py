# Check if any ID-like columns are used as numerical quantities?
# In features.py: 
# 'PassengerId' is dropped in the pipeline? 
# train_model.py:
# model.fit(X_train.drop('PassengerId', axis=1), y_train)
# Yes, it is dropped. 
# 'Num' is treated as categorical (it's added to cat_cols).
# 'Group' is added to cat_cols.
# 'Index' is added to cat_cols.
# Everything seems fine.
