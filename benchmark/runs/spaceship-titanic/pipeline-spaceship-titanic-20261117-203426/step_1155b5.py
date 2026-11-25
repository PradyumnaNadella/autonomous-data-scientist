import pandas as pd
# Check if build_features was actually called in train_model.py
with open('train_model.py', 'r') as f:
    content = f.read()
    
# Let's inspect the actual train_model.py content provided in tool call 2
# It DOES NOT actually call `build_features` on the data before fitting. 
# The prep function is defined manually in train_model.py.

print(content)
