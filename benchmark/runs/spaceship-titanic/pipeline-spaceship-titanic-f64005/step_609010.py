# Need to verify if the pipeline was used, and if leakage happened during pipeline setup.
# Actually I need to check the training script, but I don't see it in the file list.
# The user claims to have used a ColumnTransformer.
# Let's check if the pipeline logic is present.
import list_files
# I missed the training script in the file list earlier? Let me list again.
import os
print(os.listdir('.'))
