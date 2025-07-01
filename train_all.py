import subprocess, time

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

procs = []
for script in SCRIPTS:
    # wait until enough VRAM is free
    while free_vram_mb() < MIN_FREE_MB:
        time.sleep(5)
    print(f"Launching {script}…")
    p = subprocess.Popen(
        ["python", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    procs.append((script, p))

# collect and print each script's output
for script, p in procs:
    out, err = p.communicate()
    print(f"\n=== {script} stdout ===\n{out}", end="")
    if err:
        print(f"--- {script} stderr ---\n{err}", end="")

print("\nAll scripts completed.")
