# Log experiments
import json
print(json.dumps({"model": "RandomForest, default params", "cv_score": 0.7965}))
print(json.dumps({"model": "LGBM, default params", "cv_score": 0.8057}))
print(json.dumps({"model": "LGBM, n_estimators=200, learning_rate=0.05", "cv_score": 0.8062}))
