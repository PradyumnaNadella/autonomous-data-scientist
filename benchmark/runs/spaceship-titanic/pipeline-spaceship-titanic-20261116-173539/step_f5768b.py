import pandas as pd
import json

with open('experiments.jsonl', 'r') as f:
    logs = [json.loads(line) for line in f]

for log in logs:
    print(log)

# Check if Num was handled as numeric in the pipeline
# The code in train_model.py looks okay (it defines num_features),
# but let's check the build_features function specifically regarding PassengerId
# The prompt says: "Do not treat ID-like columns, or numbers extracted from them, as numeric quantities."
# The team extracted 'Group' from PassengerId and treated it as numeric.
