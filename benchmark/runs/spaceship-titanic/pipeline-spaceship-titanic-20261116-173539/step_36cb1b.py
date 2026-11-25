import json

# Log the tuned model
tuned_exp = {"model": "XGBoost, tuned (n_estimators=100, max_depth=5, learning_rate=0.1)", "cv_score": 0.8089, "notes": "Moderately tuned via RandomizedSearchCV"}

with open('experiments.jsonl', 'a') as f:
    f.write(json.dumps(tuned_exp) + '\n')
print("Tuned experiment logged.")
