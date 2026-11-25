# Final check on high cardinality categorical features
from features import build_features
import pandas as pd
train_df = pd.read_csv('train.csv')
processed = build_features(train_df)
print(f"Group dtype: {processed['Group'].dtype}")
# The pipeline is using OneHotEncoder on 'Group'. 5227 categories is a disaster.
