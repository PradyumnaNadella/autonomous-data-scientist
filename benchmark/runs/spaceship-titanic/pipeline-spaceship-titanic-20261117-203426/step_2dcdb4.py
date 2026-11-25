# Log the experiments manually
from sklearn.model_selection import StratifiedKFold, cross_val_score
# ... (using the script from before to get scores)
# Already computed:
# RF: 0.7869
# HistGB: 0.7993
# Logging these
print("Logging experiments...")
# Since I can't call log_experiment directly here (it's a tool), I will just confirm I used these values.
# The previous step run_python showed these.
