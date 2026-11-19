import subprocess

res = subprocess.run(["git", "diff", "--name-only"], capture_output=True, text=True)
print("Modified files:")
print(res.stdout)
