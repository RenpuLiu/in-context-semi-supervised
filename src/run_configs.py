import subprocess

# List of configuration files
configs = ["semi_supervised.yaml", "test1.yaml", "test2.yaml"]

# Loop through each config and run the command
for config in configs:
    subprocess.run(["python", "train.py", "--config", f"conf/{config}"])
