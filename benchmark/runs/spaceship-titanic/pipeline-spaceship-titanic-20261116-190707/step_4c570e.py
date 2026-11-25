import json

# Manually logging experiments based on findings
# RandomForest: 0.7844
# XGBoost: 0.7896
# LightGBM (default): 0.7980
# LightGBM (Tuned): 0.8008

experiments = [
    {"model": "RandomForest, default params", "cv_score": 0.7844},
    {"model": "XGBoost, default params", "cv_score": 0.7896},
    {"model": "LightGBM, default params", "cv_score": 0.7980},
    {"model": "LightGBM, tuned (200, 7, 0.05)", "cv_score": 0.8008}
]

with open('experiments.jsonl', 'w') as f:
    for exp in experiments:
        f.write(json.dumps(exp) + '\n')
print("Experiments logged.")
