# Run train_model.py
import subprocess
result = subprocess.run(['python', 'train_model.py'], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
