# Create log_experiment helper function
def log_experiment(model, cv_score, notes=None):
    print(f"EXPERIMENT: {model} | Score: {cv_score:.4f} | Notes: {notes}")

log_experiment("RandomForest, default", 0.7811, "Standard pipeline")
log_experiment("XGBoost, default", 0.7941, "Standard pipeline")
log_experiment("XGBoost, tuned", 0.7998, "n_estimators=200, max_depth=3, learning_rate=0.1")
