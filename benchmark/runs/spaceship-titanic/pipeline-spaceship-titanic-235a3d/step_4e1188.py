from features import log_experiment
log_experiment(model='Random Forest, default', cv_score=0.7869)
log_experiment(model='XGBoost, default', cv_score=0.7929)
log_experiment(model='LightGBM, default', cv_score=0.7954)
