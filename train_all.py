import subprocess
import time
import sys
import os

SCRIPTS = [
    "generate_testdata.py",  # do not comment this item out even if skipping data gen, see below
    # "training_RELU.py",
    # "inv_poly.py",
    "training_LMF.py",
    "training_LRMF.py",
    "training_MF.py",
    # "training_SORT.py",
    "RMF.py",
]

MIN_FREE_MB = 3000

def free_vram_mb():
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"
    ])
    return int(out.split()[0])

# ~~~~~~~~~ COMMENT HERE TO NOT REGENERATE DATA ~~~~~~~~~~~~
subprocess.run(["python", SCRIPTS[0]], check=True)
print("---Data generation complete---")

# 2) launch remaining scripts (fall back to CPU when GPU memory is low)
procs = []
for script in SCRIPTS[1:]:
    env = os.environ.copy()
    if free_vram_mb() < MIN_FREE_MB:
        print(f"▶ {script} (running on CPU due to low GPU memory)")
        env["CUDA_VISIBLE_DEVICES"] = ""   # force CPU in the child process
    else:
        print(f"▶ {script} (running on GPU)")
    procs.append(subprocess.Popen(["python", script], text=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  env=env))
    time.sleep(5)

# 3) wait & print their output
for p in procs:
    out, err = p.communicate()
    sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
