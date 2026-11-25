import pandas as pd
# Check if GroupSize is calculated correctly (should be per-group, not per-file)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
full = pd.concat([train.drop(columns=['Transported']), test])

full_processed = full.copy()
full_processed[['Group', 'Member']] = full_processed['PassengerId'].str.split('_', expand=True)
full_processed['GroupSize'] = full_processed.groupby('Group')['Group'].transform('count')

# The current approach in train_model.py:
def add_group_size(df):
    df = df.copy()
    df[['Group', 'Member']] = df['PassengerId'].str.split('_', expand=True)
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    return df

X_train_wrong = add_group_size(train.drop(columns=['Transported']))
X_test_wrong = add_group_size(test)

# Check if the GroupSize is the same for the same PassengerId
merged = pd.merge(X_train_wrong[['PassengerId', 'GroupSize']], full_processed[['PassengerId', 'GroupSize']], on='PassengerId', suffixes=('_split', '_full'))
print(f"Any mismatch in GroupSize calculation: {(merged['GroupSize_split'] != merged['GroupSize_full']).sum()}")
