import os, subprocess, time

SCRIPTS = [
    "generate_testdata.py",
    "training_LMF.py",
    "training_LRMF.py",
    "training_MF.py",
    "training_SORT.py",
    "training_RELU.py",
    "RMF.py",
]

MIN_FREE_MB = 1500
def free_vram_mb():
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"
    ])
    return int(out.split()[0])

for script in SCRIPTS:
    while free_vram_mb() < MIN_FREE_MB:
        time.sleep(5)
    print(f"Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    print(f"Output of {script}:\n{result.stdout}")
    if result.stderr:
        print(f"Errors in {script}:\n{result.stderr}")

print("All scripts completed.")
