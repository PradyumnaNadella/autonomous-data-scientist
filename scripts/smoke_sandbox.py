from autonomousds.sandbox.docker_runner import DockerSandbox

sb = DockerSandbox(timeout_seconds=10)

print("1. Train a model:")
r = sb.run_python("""
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
X, y = load_iris(return_X_y=True)
print("CV accuracy:", cross_val_score(LGBMClassifier(verbose=-1), X, y, cv=5).mean())
open("result.txt", "w").write("saved!")
""")
print(r.stdout, r.stderr, "ok =", r.ok)
print("Files in run folder:", sb.list_files())

print("\n2. Internet should be blocked:")
r = sb.run_python("import urllib.request; urllib.request.urlopen('https://google.com')")
print("ok =", r.ok, "(should be False)")

print("\n3. Timeout should stop long code:")
r = sb.run_python("import time; time.sleep(60)")
print("timed_out =", r.timed_out, "(should be True)")