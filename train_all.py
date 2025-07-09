import subprocess
import time
import sys

SCRIPTS = [
    "generate_testdata.py",  # do not comment this item out even if skipping data gen, see below
    "inv_poly.py",
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

# ~~~~~~~~~ COMMENT HERE TO NOT REGENERATE DATA ~~~~~~~~~~~~
# subprocess.run(["python", SCRIPTS[0]], check=True)
# print("---Data generation complete---")

# 2) launch remaining scripts when enough VRAM is free
procs = []
for script in SCRIPTS[1:]:
    while free_vram_mb() < MIN_FREE_MB:
        time.sleep(10)
    print(f"▶ {script}")
    procs.append(subprocess.Popen(["python", script], text=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE))

# 3) wait & print their output
for p in procs:
    out, err = p.communicate()
    sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
