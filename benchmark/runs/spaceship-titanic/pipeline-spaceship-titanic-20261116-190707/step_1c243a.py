# Log results
import json

def log_experiment(model_name, score, notes):
    with open('experiments.jsonl', 'a') as f:
        f.write(json.dumps({'model': model_name, 'cv_score': score, 'notes': notes}) + '\n')

log_experiment("RandomForest", 0.7903, "With all features, 5-fold CV")
log_experiment("XGBoost", 0.7970, "With all features, 5-fold CV")
log_experiment("LightGBM", 0.8063, "With all features, 5-fold CV")
