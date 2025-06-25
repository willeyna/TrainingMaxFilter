import numpy as np
# used in kmeans SDP
from sklearn.cluster import KMeans
import cvxpy as cp
from itertools import product
# unsure if this is good practice but it gets the job done
import inspect

# needed only for network training scripts 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn.functional as F
from datetime import datetime

class GroupAction:
    '''
    Takes in a group function that gives every matrix in the representation of a finite group along with
        the dimension of the representation. Also takes in the args for the given group representation.
    Stores a represention of a finite group
    Stores rep. matrices, can act on data to give orbits, can compute Frechet means, can take data in R^n and
        sramble the data considered in the quotient space given by the rep
    '''
    def __init__(self, group, dim, *args, **kwargs):
        self.group = group
        self.args = args
        self.kwargs = kwargs
        self.dim = dim
        self.custom_solver = None
        self.name = self.group.__name__
        # stores the group matrices as a |G|xdxd np array
        self.matrices = np.stack(self.group(dim, *self.args, **self.kwargs), axis = 0)
        self.order = len(self.matrices)

    def get_orbits(self, X):
        '''
        Takes in dxn dataset in R^n and returns a new |G|xdxn consisting of the
            orbit of each point under the ith elements group action as [i, :, :]
        '''
        return self.matrices@X

    def __repr__(self):
        return f"Group: {self.name}, with args [{self.args}, {self.kwargs}]."

    def frechet(self, X):
        '''
        Combinatorial brute force computation of Karcher means given a dataset X (pxn). Decently optimized :)
        '''
        d, n = X.shape
        k = self.order
        GX = self.get_orbits(X)

        # init values
        obj_value = np.inf
        aligned_data = X
        frechet_mean = None

        # For each combination (g_2, ..., g_n)
        # We choose g_1 to always be fixed to reduce computation time
        for comb in product(range(k), repeat=n-1):
            dataset_rep = GX[(0,) + comb, :, np.arange(n)].T
            u = np.mean(dataset_rep, axis=1)
            variance = np.sum((dataset_rep.T-u)**2)

            # update optimal values when a better variance is found (only store best in memory)
            if variance < obj_value:
                frechet_mean, aligned_data, obj_value = u, dataset_rep, variance

        return frechet_mean, obj_value, aligned_data

    def iterative_frechet(self, X, u = None):
        '''
        Function to iteratively find the Frechet mean of a set of n data points acted on by a group G
        Follows the "max-max" algorithm in
        "Inconsistency of Template Estimation by Minimizing of the Variance/Pre-Variance in the Quotient Space"
        Always converges in finitely many steps if G acts transitively on R^n
        '''
        d, n = X.shape
        GX = self.get_orbits(X)
        k = self.order

        # initialize frechet mean guess as a random pt in X
        if u is None:
            u = X[:, np.random.randint(n)]

        niter = 0
        while True:
            # (kxn) array where D[i,:] are the distances between each g_iX and u
            D = np.sum((GX -  u[None, :, None])**2,axis=1)  # Shape (k, n)
            # find optimal orbit point for each x_i
            closest_indices = np.argmin(D, axis=0)
            # create (dxn) array of X_i under optimal alignment wrt u
            dataset_rep = GX[closest_indices, :, np.arange(n)].T
            u_new = np.mean(dataset_rep, axis = 1)

            if np.all(u == u_new):
                break

            niter += 1
            u = u_new

        return u, dataset_rep, niter

    def frechet_gd(self, X, u=None, alpha = 1, niter = 100):
        '''
        Function to iteratively find the Frechet mean of a set of n data points acted on by a group G
        alpha = 1 is identical to IterativeFrechet
        Performs direct gradient descent on the data with fixed learning rate
        May not converge even for transitive group actions for general alpha
        '''
        d, n = X.shape
        GX = self.get_orbits(X)
        k = self.order

        # initialize frechet mean guess as a random pt in X
        if u is None:
            u = X[:, np.random.randint(n)]

        for i in range(niter):
            # (kxn) array where D[i,:] are the distances between each g_iX and u
            D = np.sum((GX -  u[None, :, None])**2,axis=1)  # Shape (k, n)
            # find optimal orbit point for each x_i
            closest_indices = np.argmin(D, axis=0)
            # create (dxn) array of X_i under optimal alignment wrt u
            dataset_rep = GX[closest_indices, :, np.arange(n)].T
            u_new = np.mean(dataset_rep, axis = 1)

            # gradF = u - u_new
            u = u - alpha*(u - u_new)

        return u, dataset_rep

    def align(self, X, x):
        '''
        Takes in a dxn dataset and returns a dxn dataset whose columns are the columns of X, x_i, acted on
            such that the distance between each x_i and x is minimized

        Note: Not called in Frechet mean methods since GX would then be computed every iteration
        '''
        GX = self.get_orbits(X)
        # (kxn) array where D[i,:] are the distances between each g_iX and x
        D = np.sum((GX -  x[None, :, None])**2,axis=1)  # Shape (k, n)
        # find optimal orbit point for each x_i
        closest_indices = np.argmin(D, axis=0)
        # create (dxn) array of X_i under optimal alignment wrt u
        aligned_data = GX[closest_indices, :, np.arange(X.shape[1])].T
        return aligned_data

    def frechet_functional(self, X, x):
        '''
        Returns the value of the Frechet functional (variance) of X at the point x
        '''
        aligned_X = self.align(X, x)
        return np.sum((aligned_X.T-x)**2)

    def dist(self, x, y):
        '''
        Takes in two (d,) arrays and computes the *squared* quotient distance
        '''

        return self.frechet_functional(x.reshape(-1,1), y)

    def dist_matrix(self, X):
        """
        Takes in dxn data matrix and returns the nxn quotient distance matrix
        Efficiently computes only the upper triangle via np broadcasting on submatrices
        """
        d, n = X.shape
        GX = self.get_orbits(X)# (k, d, n)
        D  = np.zeros((n, n))
        lower_idx = np.tril_indices(n, -1)

        for i in range(n):
            x = X[:, i]  # shape (d,)
            # broadcast gx_i...xj for j ≥ i in one go
            sq_dists = np.sum((GX[:, :, i:] - x[None, :, None])**2 ,axis=1)  # shape (k, n-i)
            D[i, i:] = np.min(sq_dists, axis=0)

        # mirror upper triangle down
        D[lower_idx] = D.T[lower_idx]
        return D


    def is_alignable(self, X, check_aligned = False):
        '''
        Determines whether a dataset is alignable
        Aligns the the first point in the dataset then iteratively checks whether quotient distances are the same
            as euclidean distances after this alignment
        Final bool input checks alignment without first trying to pre-align (checks alignment instead of alignability)
        '''
        n = X.shape[1]
        if not check_aligned:
            # align to a point, if alignable euc dist should be quotient dists
            Y = self.align(X, X[:,0])
        else:
            Y = X

        for i in range(n):
            for j in range(i+1, n):
                yi, yj = Y[:, i], Y[:, j]
                if self.dist(yi, yj) != np.sum((yi-yj)**2):
                    return False
        return True

    def randomize_reps(self, X):
        '''
        Takes in a dxn array and acts on each column by a random group element
        Returns a dxn array with ith column (g_i * x_i)
        '''
        n = X.shape[1]
        GX = self.get_orbits(X)
        gi = np.random.randint(0, self.order, n)
        dataset_rep = GX[gi, :, np.arange(n)].T

        return dataset_rep

    def kmeans(self, X, k, init = None, solver = 'iterative', max_iters=100, tol=1e-4):
        """
        Frechet kmeans using Lloyds algorithm
        X: array of shape (d, n)
        k: number of clusters
        returns: centroids C (d×k), and labels (n,)
        """
        # init centroids by choosing k random points from X
        if init is None:
            C = X[:, np.random.choice(X.shape[1], k, replace=False)]
        else:
            C = init

        for i in range(max_iters):
            dlist = []
            for c in C.T:
                Y = self.align(X, c)
                dlist.append(np.sum((Y - c[:,None])**2, axis=0))
            labels = (np.stack(dlist)).argmin(axis=0)

            # *** determine new centroids using a frechet means solver ***
            columns = []
            for j in range(k):
                if solver == 'exact':
                    col = self.frechet(X[:, labels == j])[0]
                elif solver == 'iterative':
                    col = self.iterative_frechet(X[:, labels == j])[0]
                elif solver == 'custom':
                    col = self.custom_solver(X[:, labels == j])
                columns.append(col)
            new_C = np.column_stack(columns)

            # check convergence using quotient dist
            if np.sum([self.dist(C[:,i], new_C[:,i]) for i in range(k)]) < tol:
                break
            C = new_C

        return C, labels

    def kmeans_sdp(self, X, k, solver = 'iterative'):
        """
        Finds the Frechet kMeans using quotient distance and the kMeans Pei-Wei SDP
        X:       array (d × n)
        k:       number of clusters

        Returns:
          C      – final centroids in original space (d×k)
          labels – cluster assignments (n,)
        (Does not return Z_opt)
        """
        d, n = X.shape

        # Build pairwise kernel matrix K = -D (D dist matrix (n×n) through the quotient distance)
        K = -self.dist_matrix(X)

        # Set up SDP variable & constraints
        Z = cp.Variable((n, n), symmetric=True)
        constr = [
            Z >> 0,                # Z ⪰ 0
            Z >= 0,                # entrywise ≥ 0
            Z @ np.ones(n) == 1,   # Z·1 = 1
            cp.trace(Z) == k       # trace(Z) = k
        ]

        # Objective: max Tr(K Z)
        prob = cp.Problem(cp.Maximize(cp.trace(K @ Z)), constr)
        prob.solve(solver='SCS', verbose=False)

        Z_opt = Z.value
        if Z_opt is None:
            raise RuntimeError("SDP failed to find a solution")

        # Spectral rounding: embed via top-k eigenspace
        vals, vecs = np.linalg.eigh(Z_opt)
        topk = np.argsort(vals)[-k:]
        U = vecs[:, topk] * np.sqrt(vals[topk])
        U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)
        embed_labels = KMeans(n_clusters=k).fit_predict(U_norm)

        ## Computing *approximate* final cluster means (does not affect clustering)
        columns = []
        for j in range(k):
            if solver == 'exact':
                col = self.frechet(X[:, embed_labels == j])[0]
            elif solver == 'iterative':
                col = self.iterative_frechet(X[:, embed_labels == j])[0]
            elif solver == 'custom':
                col = self.custom_solver(X[:, labels == j])[0]
            columns.append(col)

        C = np.column_stack(columns)

        return C, embed_labels

class CustomFrechet_GroupAction(GroupAction):
    '''
    Same as GroupAction but requires frechet_solver, a custom function that takes a (dxn) data matrix and returns
        the frechet mean relative to the specified group action. Allows for group specific SDP/spectral solvers.
    '''
    def __init__(self, group, dim, frechet_solver, *args, **kwargs):
        super().__init__(group, dim, *args, **kwargs)
        self.custom_solver = frechet_solver

import torch

class GPU_GroupAction(GroupAction):
    '''
    A subclass of GroupAction that uses PyTorch for efficient GPU-based computations.
    '''
    def __init__(self, group, dim, device, *args, **kwargs):
        super(GPU_GroupAction, self).__init__(group, dim, *args, **kwargs)
        # device to run tensor computations on
        self.device = device or torch.device('cpu')  # fallback if not provided

        # Convert the group matrices to torch tensors for GPU compatibility
        self.matrices = torch.stack([torch.tensor(mat, dtype=torch.float32, device=self.device) for mat in self.group(dim, *self.args,**self.kwargs)], dim=0)

    def dist_matrix(self, X):
        """
        Takes in a (d, n) data TENSOR and returns an (n, n) quotient distance matrix.
        Efficiently computes only the upper triangle using broadcasting.
        """
        d, n = X.shape
        GX = self.get_orbits(X)  # (k, d, n), assumed to be torch.Tensor
        D = torch.zeros((n, n), device=self.device)

        for i in range(n):
            x = X[:, i]  # (d,)
            # Broadcast subtraction: (k, d, n - i)
            diff = GX[:, :, i:] - x.view(1, -1, 1)
            sq_dists = (diff ** 2).sum(dim=1)  # (k, n - i)
            D[i, i:] = sq_dists.min(dim=0).values

        # Reflect upper triangle to lower triangle
        idx_lower = torch.tril_indices(n, n, offset=-1, device=X.device)
        D[idx_lower[0], idx_lower[1]] = D[idx_lower[1], idx_lower[0]]
        return D


# ─── Group matrix constructors ────────────────────────────────────────────────
# functions that generate list of all matrices in a finite group

def cyclic_translations(d):
    '''
    Returns the d cyclic permutation matrices
    '''
    C = np.eye(d)[:, np.append(np.arange(1, d), 0)]
    return [np.linalg.matrix_power(C, j) for j in range(d)]

def pmId(d):
    '''
    Z2 acting as sign change in every dimension
    '''
    I = np.eye(d)
    return [I, -I]

def reflection(d, axis=None):
    '''
    Householder reflection matrix about hyperplane perp to the given axis
    '''
    if axis is None:
        axis = np.zeros(d)
        axis[-1] = 1
    R = np.eye(d) - 2 * np.outer(axis, axis) / np.dot(axis, axis)
    return [np.eye(d), R]

def rotations(d, orders):
    '''
    Gives an element of SO(d) in canonical form by order d (finite)
    rotations on the 2x2 subspaces
    '''
    m = d // 2
    R = np.eye(d)
    for i, n in enumerate(orders[:m]):
        if n > 1:
            theta = 2 * np.pi / n
            R[2*i:2*i+2, 2*i:2*i+2] = [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)],
            ]
    max_power = np.lcm.reduce([n for n in orders[:m] if n > 1], initial=1)
    return [np.linalg.matrix_power(R, k) for k in range(max_power)]

def diagonal_sign(d):
    """
    All 2^d diagonal matrices with ±1 on the diagonal. Reflection group
    """
    mats = []
    for signs in product([1, -1], repeat=d):
        mats.append(np.diag(signs))
    return mats

def even_diagonal_sign(d):
    """
    Diagonal ±1 matrices with an even number of −1 entries,
    i.e. det = +1.  Size = 2^(d-1).
    NOT a reflection group, Z2 acting on 2 dim subspaces by +-Id
    """
    mats = []
    for signs in product([1, -1], repeat=d):
        if sum(s == -1 for s in signs) % 2 == 0:
            mats.append(np.diag(signs))
    return mats


def dihedral_group(d, n, axes=(0, 1)):
    """
    Action of D_n (dihedral group) on R^d by acting on the 2D subspace spanned by
    the first two dimensions. Acts as the natrual Dihedral group action on R^2
    """
    i, j = axes
    mats = []

    for k in range(n):
        θ = 2 * np.pi * k / n
        # rotation in (i,j) plane
        R = np.eye(d)
        R[np.ix_((i, j), (i, j))] = [[np.cos(θ), -np.sin(θ)],
                                     [np.sin(θ),  np.cos(θ)]]
        mats.append(R)

        # reflection: flip j-axis after rotating
        F = R.copy()
        F[j, j] *= -1
        mats.append(F)

    return mats


# ─── Utility functions ────────────────────────────────────────────────

# N points in R^p uniform in B(y,R)
def sample_unif_ball(n, d, R=1, y=None):

    if y is None:
        y = np.zeros(d)

    # Ensure y is an array and reshape it for broadcasting
    y = np.array(y).reshape(-1, 1)  # Reshape y to (p, 1)

    # Generate random directions
    directions = np.random.randn(d, n)
    directions /= np.linalg.norm(directions, axis=0)

    # Generate random distances from the center within [0, R]
    distances = np.random.rand(n) ** (1 / d) * R

    # Scale directions by distances and add the center point y
    points = y + directions * distances
    return points

# goemans-williamson SDP used for SDP computation of +-Id Frechet mean
def GW_SDP(X):
    ############ ChatGPT coded
    N = X.shape[1]

    # Define the variable Z as a p by p matrix
    Z = cp.Variable((N, N), symmetric=True)

    # Define the objective: maximize <X^T X, Z>
    objective = cp.Maximize(cp.trace((X.T @ X) @ Z))

    # Define the constraints: Z is positive semidefinite and diagonal is all ones
    constraints = [Z >> 0,  # Z is positive semidefinite
                   cp.diag(Z) == np.ones(N)]  # Diagonal of Z is all ones

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    # The optimal value of Z
    Z_opt = Z.value
    #############

    return Z_opt

def max_filter(templates, X_orbits):
    '''
    X_orbits kxdxn data array
    templates txd matrix each row is a template

    returns max-filter augmented data (torch tensor)
    '''
    inner_products = torch.einsum('td,kdn->ktn', templates, X_orbits)
    maxes, argmaxes = torch.max(inner_products, axis=0)
    return maxes

def sorted_filter(templates, X_orbits):
    """
    templates: (t, d)
    X_orbits:  (k, d, n)
    returns:   (k*d, n)
    """
    # inner products → (t, k, n)
    inner_products = torch.einsum('td,kdn->ktn', templates, X_orbits)
    sorted_products, _ = torch.sort(inner_products, dim=0)

    return sorted_products.reshape(-1, X_orbits.size(2))

def squared_lipschitz(squared_distance, fX):
    """
    Compute the squared Lipschitz constants (alpha_squared, beta_squared)
    for a mapping fX given the original squared-distance matrix.
    """
    # 1) Pairwise squared distances in feature space
    fxT = fX.T                                  # (n, target_dim)
    diff = fxT.unsqueeze(1) - fxT.unsqueeze(0)  # (n, n, target_dim)
    fx_sq_dist = (diff ** 2).sum(dim=-1)        # (n, n)

    # 2) Select only unique i < j entries (upper triangle mask)
    mask = torch.triu(torch.ones_like(squared_distance), diagonal=1).bool()
    orig_sq   = squared_distance[mask]          # (n*(n-1)/2,)
    mapped_sq = fx_sq_dist[mask]                # (n*(n-1)/2,)

    # 3) Compute expansion factors safely
    expansions = mapped_sq / orig_sq

    # 4) Return min and max
    alpha_squared = expansions.min()
    beta_squared  = expansions.max()
    return alpha_squared, beta_squared

# chatgpt made-- used for training RELU non-invariant network
def mask_ignore_block_diagonals(n, k, device=None):
    nk = n * k
    idx = torch.arange(nk, device=device)
    # base mask: False when in same block diagonal (i % n == j % n)
    base_mask = (idx.unsqueeze(1) % n) != (idx.unsqueeze(0) % n)
    # additionally mask out the upper diagonal (i < j)
    upper_diagonal_mask = idx.unsqueeze(1) >= idx.unsqueeze(0)
    return base_mask & upper_diagonal_mask
