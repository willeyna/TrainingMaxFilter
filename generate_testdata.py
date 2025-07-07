import numpy as np

#Create a random test set n pts in d dim
n_test = 2000
d = 2
X_test = np.random.normal(0, 1, size=(d, n_test))
np.save("X_test.npy", X_test)
