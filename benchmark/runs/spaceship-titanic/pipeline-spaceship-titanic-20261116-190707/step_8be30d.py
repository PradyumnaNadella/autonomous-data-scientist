import pandas as pd
experiments = pd.read_json('experiments.jsonl', lines=True)
print(experiments)
