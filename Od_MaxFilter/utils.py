import subprocess

# List of scripts to run in order
scripts = [
    "generate_testdata.py",
    "training_LMF.py",
    "training_LRMF.py",
    "training_MF.py",
    "training_SORT.py",
    "training_RELU.py",
    "RMF.py"
]

# Run each script one by one
for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)

    # Print the output and any errors
    print(f"Output of {script}:\n{result.stdout}")
    if result.stderr:
        print(f"Errors in {script}:\n{result.stderr}")
