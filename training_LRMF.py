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
X_test_orbits = G.get_orbits(X_test)
D_test = G.dist_matrix(X_test)

batch_size = 20 # 20 works well for pmId d=3
n_trials = 1 #The number of times we will train a new model from scratch
n_epochs = 100 #The number of training epochs for each model
grad_steps_per_epoch = 200 #The number of gradient descent iterations in each training epoch
lr = 1e-2 # learning rate default is 1e-3
lr_period =  n_epochs//5
######################################################## TRAINING

all_test_distortions = []
all_trained_distortions = []
all_trained_templates = []
all_trained_Ls = []

for trial in range(n_trials):
    test_distortions = []
    train_distortions = []

    # max filter template layer
    templates = torch.normal(0, 1, (t, d), requires_grad=True, device=device)
    # linear layer
    L = torch.normal(0, 1, (target_dim, t), requires_grad=True, device=device)

    optimizer = torch.optim.Adam([L, templates], lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, lr_period)

    for epoch in range(n_epochs):
        for step in range(grad_steps_per_epoch):
            optimizer.zero_grad()

            # Sample training batch
            X =  torch.normal(0, 1, (d, batch_size), device=device)
            D = G.dist_matrix(X)           # Tensor (n, n)
            X_orbits = G.get_orbits(X)     # Tensor (k, d, n)

            # Forward pass
            norm_templates = F.normalize(templates, dim=1)
            filter_features = max_filter(norm_templates, X_orbits)
            features = L @ filter_features

            alpha_sq, beta_sq = squared_lipschitz(D, features)
            loss = beta_sq / alpha_sq

            loss.backward()
            optimizer.step()

        # Test each epoch
        print(f"Epoch {epoch}", end='\r')
        with torch.no_grad():
            # Compute training distortion on last batch of training data (X, D, etc.)
            norm_templates = F.normalize(templates, dim=1)
            train_features = max_filter(norm_templates, X_orbits)
            train_features = L @ train_features
            alpha_train_sq, beta_train_sq = squared_lipschitz(D, train_features)
            distortion_train = (beta_train_sq / alpha_train_sq).item()

            # Compute test distortion
            test_features = max_filter(norm_templates, X_test_orbits)
            test_features = L @ test_features
            alpha_test_sq, beta_test_sq = squared_lipschitz(D_test, test_features)
            distortion_test = (beta_test_sq / alpha_test_sq).item()

            train_distortions.append(distortion_train)
            test_distortions.append(distortion_test)
        scheduler.step()

    all_trained_distortions.append(train_distortions)
    all_test_distortions.append(test_distortions)
    all_trained_templates.append(norm_templates.detach().cpu().numpy())
    all_trained_Ls.append(L.detach().cpu().numpy())

avg_final_error = np.mean([d[-1] for d in all_test_distortions])

fig = plt.figure(figsize = (15,5))
for distortion_function in all_test_distortions:
    plt.plot(distortion_function, alpha=0.3, c='red')

# for i, distortion_function in enumerate(all_trained_distortions):
#     if i ==0:
#         plt.plot(distortion_function, alpha=0.1, c='blue', label = 'Training Error')
#     else:
#         plt.plot(distortion_function, alpha=0.1, c='blue')

textstr = '\n'.join([
    str(G).split(',')[0],
    f'Input Data Dimension: {d}',
    f'Embedding Dimension: {target_dim}',
    f'Final Mean Squared Distortion: {avg_final_error:.2f}',
    f'Batch size: {batch_size}',
    f'Grad steps/epoch: {grad_steps_per_epoch}',
    f'Learning rate: {lr}',
])
# Place text box in upper right in axes coords
props = dict(boxstyle='round', alpha=0.1)
plt.gca().text(0.98, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=props)


plt.title("Trained Linear(Random Max Filter(X)) Model Test Error")
plt.xlabel("Epoch")
plt.ylabel("Squared Distortion")

# Get current time formatted as YYYY-MM-DD_HH-MM
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
# time-based label for now since all params are shown in the figure
plt.savefig(f'./Results/LRMF_{timestamp}.png')
