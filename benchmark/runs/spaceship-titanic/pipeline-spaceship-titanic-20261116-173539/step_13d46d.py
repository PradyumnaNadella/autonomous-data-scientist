import pandas as pd
# Check if preprocessor fits on train only
# The team code:
# preprocessor = ColumnTransformer([...])
# clf = Pipeline([('prep', preprocessor), ('clf', ...)])
# clf.fit(train_df.drop(...), ...)
# This is correct.
# Wait, check if they used build_features on both independently before passing to the pipeline.
# train_df = build_features(train_df)
# test_df = build_features(test_df)
# This is ALSO correct, because build_features does not calculate global stats.

# One issue: 'Group' is treated as a numerical input in the 'num' branch of the ColumnTransformer.
# This violates "Do not treat ID-like columns... as numeric quantities."
# Group is an ID-like identifier (it represents which group the passenger belongs to), not a magnitude.
