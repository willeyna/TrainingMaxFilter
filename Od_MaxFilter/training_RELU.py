from groupy import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################## PARAMETERS
# load test data so that it is the same for every model
X_test_np = np.load('X_test.npy')
X_test = torch.from_numpy(X_test_np).float().to(device)
# number of templates in max filter
d = X_test.shape[0]
t = 3*d
# currently maintain dimensionality given by max filtering
target_dim = t

G = GPU_GroupAction(cyclic_translations, d, device=device)
k = G.order
X_test_orbits = G.get_orbits(X_test)
D_test = G.dist_matrix(X_test)
D_test_expanded = D_test.repeat(k,k)

batch_size = 20 # Number of randomly generated samples per minibatch
n_trials = 5 #The number of times we will train a new model from scratch
n_epochs = 100 #The number of training epochs for each model
grad_steps_per_epoch = 200 #The number of gradient descent iterations in each training epoch
lr = 1e-2 # learning rate (default is 1e-3 for ADAM)
lr_period = n_epochs//5 # period for cosine annealing
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
            alpha_sq, beta_sq = squared_lipschitz(D_expanded, features, k, on_orbits = True)
            loss = beta_sq / alpha_sq

            loss.backward()
            optimizer.step()

        # Evaluation
        print(f"Epoch {epoch}", end='\r')
        with torch.no_grad():
            # Train features
            hidden_train = F.relu(W @ X_orbits_reshaped)
            features_train = L @ hidden_train
            alpha_train_sq, beta_train_sq = squared_lipschitz(D_expanded, features_train, k, on_orbits=True)
            distortion_train = (beta_train_sq / alpha_train_sq).item()

            # Test
            X_test_orbits_reshaped = (X_test_orbits.permute(1, 0, 2).reshape(d, k*X_test_orbits.shape[2]))
            hidden_test = F.relu(W @ X_test_orbits_reshaped)
            features_test = L @ hidden_test

            alpha_test_sq, beta_test_sq = squared_lipschitz(D_test_expanded, features_test, k, on_orbits = True)
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
