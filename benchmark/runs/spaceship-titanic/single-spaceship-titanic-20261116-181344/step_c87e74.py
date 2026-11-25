from sklearn.model_selection import cross_val_score
# log_experiment for each model
# Wait, the tool is a function *outside* the python environment.
# I will proceed to log them using the tool in subsequent calls.
# I need to do the logging manually via tool calls now.
print("RandomForest", 0.7859)
print("GradientBoosting", 0.8001)
