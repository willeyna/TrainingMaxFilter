# Training Max Filter

This repository contains experimental code for training max-filter-based embeddings for orbit spaces under group actions to attain low metric distortion. It includes PyTorch training scripts, utilities for finite and continuous group actions, notebooks for visualization, sample shape data, congressional district shapefiles, and saved experiment outputs.

## Related Paper

This code is related to the paper:

[https://arxiv.org/pdf/2603.23643](https://arxiv.org/pdf/2603.23643)

and further numerical details can be found in `Bilipschitz_Training.pdf`.

## Repository Contents

- `groupy.py` - core utilities for group actions, quotient distances, max filters, Lipschitz/distortion calculations, and related helper functions.
- `training_LMF.py`, `training_LRMF.py`, `training_MF.py`, `training_RELU.py`, `training_SORT.py` - training scripts for different embedding/filter variants.
- `RMF.py` - evaluates randomly initialized max filters as a baseline.
- `generate_testdata.py` - creates a default `X_test.npy` dataset used by the training and evaluation scripts.
- `train_all.py` - launches several training/evaluation scripts, using GPU when enough CUDA memory is available and falling back to CPU otherwise.
- `Results/` - generated plots from experiments.

- Most experiment settings are configured near the top of each script, including the group action, number of templates, batch size, number of trials, number of epochs, and learning rate.
- Different training script lines are needed when considering continuous group actions (i.e. orthogonal or rotations) rather than finite ones. 
