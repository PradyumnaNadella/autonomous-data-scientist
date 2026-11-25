import json

# Log experiments as required
experiments = [
    {"model": "Random Forest, default params", "cv_score": 0.7936, "notes": "Standard preprocessing, Group as categorical."},
    {"model": "XGBoost, default params", "cv_score": 0.7945, "notes": "Standard preprocessing, Group as categorical."}
]

with open('experiments.jsonl', 'w') as f:
    for exp in experiments:
        f.write(json.dumps(exp) + '\n')
print("Experiments logged.")
