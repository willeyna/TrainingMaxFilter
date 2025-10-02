import numpy as np
import torch

#Create a random test set n pts in d dim
n_test = 2000
d = 2

# real vector input data
# X_test = np.random.normal(0, 1, size=(d, n_test))
# complex input data for shape space
# X_test = torch.randn((d, n_test), dtype=torch.complex64)
# real matrices
p = 2
X_test = np.random.normal(0, 1, size=(d, p, n_test))
np.save("X_test.npy", X_test)
