import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from groupy import *
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def max_filter(templates, X_orbits):
    '''
    X_orbits kxdxn data array
    templates txd matrix each row is a template

    returns max-filter augmented data (torch tensor)
    '''
    inner_products = torch.einsum('td,kdn->ktn', templates, X_orbits)
    maxes, argmaxes = torch.max(inner_products, axis=0)
    return maxes

# chatgpt made
def mask_ignore_block_diagonals(n, k, device=None):
    nk = n * k
    idx = torch.arange(nk, device=device)
    # base mask: False when in same block diagonal (i % n == j % n)
    base_mask = (idx.unsqueeze(1) % n) != (idx.unsqueeze(0) % n)
    # additionally mask out the upper diagonal (i < j)
    upper_diagonal_mask = idx.unsqueeze(1) >= idx.unsqueeze(0)
    return base_mask & upper_diagonal_mask

# ignores distance-0 data points in computing bounds since they kill everything.
def squared_lipschitz_orbits(squared_distance, fX, k):
    """
    Compute the squared Lipschitz constants (alpha_squared, beta_squared)
    for a mapping fX given the original squared-distance matrix.
    """
    # 1) Pairwise squared distances in feature space
    fxT = fX.T                                  # (n, target_dim)
    diff = fxT.unsqueeze(1) - fxT.unsqueeze(0)  # (n, n, target_dim)
    fx_sq_dist = (diff ** 2).sum(dim=-1)        # (n, n)

    # 2) Select only unique i < j entries (upper triangle mask)
    mask = mask_ignore_block_diagonals(squared_distance.shape[0]//k, k)
    orig_sq   = squared_distance[mask]          # (n*(n-1)/2,)
    mapped_sq = fx_sq_dist[mask]                # (n*(n-1)/2,)

    # 3) Compute expansion factors safely
    expansions = mapped_sq / orig_sq

    # 4) Return min and max
    alpha_squared = expansions.min()
    beta_squared  = expansions.max()
    return alpha_squared, beta_squared

######################################################## PARAMETERS
# load test data so that it is the same for every model
X_test_np = np.load('X_test.npy')
X_test = torch.from_numpy(X_test_np).float().to(device)
# number of templates in max filter
d = X_test.shape[0]
t = 3*d
# currently maintain dimensionality given by max filtering
target_dim = t

G = GPU_GroupAction(pmId, d, device=device)
k = G.order
X_test_orbits = G.get_orbits(X_test)
D_test = G.dist_matrix(X_test)
D_test_expanded = D_test.repeat(k,k)

batch_size = 20 # 20 works well for pmId d=3
n_trials = 10 #The number of times we will train a new model from scratch
n_epochs = 100 #The number of training epochs for each model
grad_steps_per_epoch = 200 #The number of gradient descent iterations in each training epoch
lr = 1e-2 # learning rate default is 1e-3
lr_period = n_epochs//5
######################################################## TRAINING

all_test_distortions = []
all_trained_distortions = []
all_trained_Ws = []
all_trained_Ls = []

hidden_dim = k*target_dim
for trial in range(n_trials):
    test_distortions = []
    train_distortions = []

    # Initialize weights with good scaling
    W = torch.normal(0, 1, (hidden_dim, d), requires_grad=True, device=device)
    L = torch.normal(0, 1, (target_dim, hidden_dim), requires_grad=True, device=device)

    optimizer = torch.optim.Adam([W, L], lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, lr_period)

    for epoch in range(n_epochs):
        for step in range(grad_steps_per_epoch):
            optimizer.zero_grad()

            # Sample training batch
            X = torch.normal(0, 1, (d, batch_size), device=device)
            D = G.dist_matrix(X)            # shape: (n, n), squared distances
            # create k*n by k*n distance matrix for spun up data-set
            D_expanded = D.repeat(k,k)
            X_orbits = G.get_orbits(X)      # shape: (k, d, n)

            # Reshape orbits into separate samples
            X_orbits_reshaped = (X_orbits.permute(1, 0, 2).reshape(d, k*X_orbits.shape[2]))  # (d, k*n)

            # Forward pass
            hidden = F.relu(W @ X_orbits_reshaped)        # (hidden_dim, k*n)
            features = L @ hidden                         # (target_dim, k*n)

            # Lipschitz loss
            alpha_sq, beta_sq = squared_lipschitz_orbits(D_expanded, features, k)
            loss = beta_sq / alpha_sq

            loss.backward()
            optimizer.step()

        # Evaluation
        print(f"Epoch {epoch}", end='\r')
        with torch.no_grad():
            # Train features
            hidden_train = F.relu(W @ X_orbits_reshaped)
            features_train = L @ hidden_train
            alpha_train_sq, beta_train_sq = squared_lipschitz_orbits(D_expanded, features_train, k)
            distortion_train = (beta_train_sq / alpha_train_sq).item()

            # Test
            X_test_orbits_reshaped = (X_test_orbits.permute(1, 0, 2).reshape(d, k*X_test_orbits.shape[2]))
            hidden_test = F.relu(W @ X_test_orbits_reshaped)
            features_test = L @ hidden_test

            alpha_test_sq, beta_test_sq = squared_lipschitz_orbits(D_test_expanded, features_test, k)
            distortion_test = (beta_test_sq / alpha_test_sq).item()

            train_distortions.append(distortion_train)
            test_distortions.append(distortion_test)

        scheduler.step()

    all_trained_distortions.append(train_distortions)
    all_test_distortions.append(test_distortions)
    all_trained_Ws.append(W.detach().cpu().numpy())
    all_trained_Ls.append(L.detach().cpu().numpy())

avg_final_error = np.mean([d[-1] for d in all_test_distortions])

fig = plt.figure(figsize = (15,5))
for distortion_function in all_test_distortions:
    plt.plot(distortion_function, alpha=0.3, c='red')


textstr = '\n'.join([
    str(G).split(',')[0],
    f'Input Data Dimension: {d}',
    f'Embedding Dimension: {target_dim}',
    f'Final Mean Distortion: {avg_final_error:.2f}',
    f'Batch size: {batch_size}',
    f'Grad steps/epoch: {grad_steps_per_epoch}',
    f'Learning rate: {lr}',
])
# Place text box in upper right in axes coords
props = dict(boxstyle='round', alpha=0.1)
plt.gca().text(0.98, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=props)


plt.title("Trained Linear(RELU(Linear(X))) Model Test Error")
plt.xlabel("Epoch")
plt.ylabel("Squared Distortion")
plt.yscale('log')

# Get current time formatted as YYYY-MM-DD_HH-MM
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
# time-based label for now since all params are shown in the figure
plt.savefig(f'./Results/RELU_{timestamp}.png')
