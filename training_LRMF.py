from groupy import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################## PARAMETERS
# load test data so that it is the same for every model
X_test_np = np.load('X_test.npy')
X_test = torch.from_numpy(X_test_np).float().to(device)
# number of templates in max filter
d,n = X_test.shape
t = 3*d
# currently maintain dimensionality given by max filtering
target_dim = t

G = GPU_GroupAction(cyclic_translations, d, device=device)
k = G.order
X_test_orbits = G.get_orbits(X_test)

batch_size = 20 # Number of randomly generated samples per minibatch
n_trials = 10 # The number of times we will train a new model from scratch
n_epochs = 100 # The number of training epochs for each model
grad_steps_per_epoch = 200 # The number of gradient descent iterations in each training epoch
lr = 1e-2 # learning rate (default is 1e-3 for ADAM)
lr_period = n_epochs # period for cosine annealing
block_size = 1000 # how many data points to include in each test distance matrix
######################################################## TRAINING

all_test_distortions = []
all_trained_distortions = []
all_trained_templates = []
all_trained_Ls = []

for trial in range(n_trials):
    print("Trial", trial)
    test_distortions = []
    train_distortions = []

    # max filter template layer
    templates = torch.normal(0, 1, (t, d), device=device)
    # linear layer
    L = torch.normal(0, 1, (target_dim, t), requires_grad=True, device=device)

    optimizer = torch.optim.Adam([L], lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, lr_period)

    for epoch in range(n_epochs):
        for step in range(grad_steps_per_epoch):
            optimizer.zero_grad()

            # Sample training batch
            X =  torch.normal(0, 1, (d, batch_size), device=device)
            # full distance matrix for minibatch-- important to keep mini!
            D = G.dist_matrix(X)           # Tensor (n, n)
            X_orbits = G.get_orbits(X)     # Tensor (k, d, n)

            # Forward pass
            norm_templates = F.normalize(templates, dim=1)
            filter_features = max_filter(norm_templates, X_orbits)
            features = L @ filter_features

            DfX = torch.cdist(features.T, features.T)**2
            alpha_sq, beta_sq = squared_lipschitz(D, DfX)
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
            DfX_train = torch.cdist(train_features.T, train_features.T)**2
            alpha_train_sq, beta_train_sq = squared_lipschitz(D, DfX_train)
            distortion_train = (beta_train_sq / alpha_train_sq).item()

            # Compute test distortion ---
            # initalize alpha and beta
            alpha_test_sq = torch.tensor(float("inf"), device=device)
            beta_test_sq  = torch.tensor(0, device=device)
            # break test set into blocks over which to compute distance matrices
            for i in range(0, n, block_size):
                # choose subset of x_i and f(x_i)
                Xi = X_test[:, i:i+block_size]
                fXi = L @ max_filter(norm_templates, X_test_orbits[:, :, i:i+block_size])
                for j in range(i, n, block_size):
                    Xj = X_test[:, j:j+block_size]
                    fXj = L @ max_filter(norm_templates, X_test_orbits[:, :, j:j+block_size])
                    # compute block distance matrices
                    # when Xi=Xj function automatically only computes n choose 2 distances
                    DX_ij   = G.dist_matrix(Xi, Xj)
                    DfX_ij  = torch.cdist(fXi.T, fXj.T)**2
                    # compute constants over that block
                    alpha_ij, beta_ij = squared_lipschitz(DX_ij, DfX_ij)
                    # update either lipschitz constant if a worse one is found
                    alpha_test_sq = torch.minimum(alpha_test_sq, alpha_ij)
                    beta_test_sq = torch.maximum(beta_test_sq, beta_ij)

            distortion_test = (beta_test_sq / alpha_test_sq).item()

            train_distortions.append(distortion_train)
            test_distortions.append(distortion_test)
        scheduler.step()

    all_trained_distortions.append(train_distortions)
    all_test_distortions.append(test_distortions)
    all_trained_templates.append(norm_templates.detach().cpu().numpy())
    all_trained_Ls.append(L.detach().cpu().numpy())

mean_final_error = np.mean([d[-1] for d in all_test_distortions])
median_final_error = np.median([d[-1] for d in all_test_distortions])

fig = plt.figure(figsize = (15,5))
for distortion_function in all_test_distortions:
    plt.plot(distortion_function, alpha=0.3, c='red')

textstr = '\n'.join([
    str(G).split(',')[0],
    f'Input Data Dimension: {d}',
    f'Embedding Dimension: {target_dim}',
    f'Batch size: {batch_size}',
    f'Grad steps/epoch: {grad_steps_per_epoch}',
    f'Learning rate: {lr}',
    f'Mean Final Squared Distortion: {mean_final_error:.2f}',
    f'Median Final Squared Distortion: {median_final_error:.2f}'
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
