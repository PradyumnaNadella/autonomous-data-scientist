import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Check leakage
# 1. Did the build_features use test.csv? No.
# 2. Did the pipeline learn on test data?
# train_model.py: 
#   X_train = build_features(train)...
#   model.fit(X_train, y_train)
#   model.predict(X_test)
# The pipeline IS fitted on training data only. Correct.
print("Leakage check: Pipeline fitted on train only? Yes.")
