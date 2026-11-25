from sklearn.model_selection import cross_val_score, StratifiedKFold
# Using the scores from the previous step
# rf_score: 0.7861664658208731
# gb_score: 0.7975274762217545

# Log experiments
from sklearn.model_selection import cross_val_score
# I need to call log_experiment, but I'm in a fresh process, so I need to re-import
# Actually, the log_experiment is a provided tool, I don't import it.

# Let's re-run the logging. Wait, I cannot re-run cross_val_score without defining the pipeline again.
# That's fine, I have the scores.
pass
