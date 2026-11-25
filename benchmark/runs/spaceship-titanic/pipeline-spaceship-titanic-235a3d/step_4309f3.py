# The team's reported importance:
# 1. num__VRDeck
# 2. num__Age
# 3. num__FoodCourt
# 4. num__Spa
# 5. num__RoomService
# These are all numeric features. This is plausible.

# Are there any issues?
# They didn't show the code for the pipeline in their files.
# They didn't show the feature importance calculation.
# I will check if their predictions.csv is correctly formatted.

df_pred = pd.read_csv('predictions.csv')
print(df_pred.columns)
print(df_pred.head())
print(df_pred.dtypes)
