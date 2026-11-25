import numpy as np

# Check feature importances
# Access the classifier from the pipeline
classifier = final_model.named_steps['classifier']
# Access the preprocessor
preprocessor = final_model.named_steps['preprocessor']
# Get feature names after one-hot encoding
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_names = cat_encoder.get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_names)

importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 5 features:")
for i in range(5):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]}")
