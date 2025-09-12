from groupy import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.manual_seed(2)
# torch.cuda.manual_seed_all(2)
######################################################## PARAMETERS
# load test data so that it is the same for every model
X_test_np = np.load('district_outlines.npy')
X_test = torch.from_numpy(X_test_np).to(torch.complex64).to(device)
# number of templates in max filter
d,n = X_test.shape
t = 100
# currently maintain dimensionality given by max filtering
target_dim = t

batch_size = 128 # Number of randomly generated samples per minibatch
n_trials = 1 # The number of times we will train a new model from scratch
n_epochs = 3000 # The number of training epochs for each model
grad_steps_per_epoch = np.ceil(n/batch_size) # The number of gradient descent iterations in each training epoch
lr = 5e-3 # learning rate (default is 1e-3 for ADAM)
lr_period = n_epochs # period for cosine annealing
######################################################## TRAINING

all_test_distortions = []
all_trained_distortions = []
all_trained_templates = []
all_trained_Ls = []

### Set up training data (districts)

# train data is also test data here
train_dataset = torch.utils.data.TensorDataset(X_test.T)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for trial in range(n_trials):
    print()
    print("Trial", trial)
    test_distortions = []
    train_distortions = []

    # max filter template layer; complex Gaussian
    templates = torch.randn((t, d), dtype=torch.cfloat, device=device, requires_grad=True)
    # linear layer
    L = torch.normal(0, 1, (target_dim, t), requires_grad=True, device=device)

    optimizer = torch.optim.Adam([templates, L], lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, lr_period)

    for epoch in range(n_epochs):
        for batch in train_loader:
            optimizer.zero_grad()

            X = batch[0].T
            # Sample training batch
            # X = torch.randn((d, batch_size), dtype=torch.cfloat, device=device)
            # full distance matrix for minibatch-- important to keep mini!
            D = shape_dist_matrix(X)           # Tensor (n, n)

            # Forward pass
            norm_templates = F.normalize(templates, dim=1)
            filter_features = shape_max_filter(norm_templates, X)
            features = L @ filter_features

            DfX = torch.cdist(features.T, features.T)
            alpha, beta = lipschitz(D, DfX)

            loss = beta / alpha

            loss.backward()
            torch.nn.utils.clip_grad_norm_([templates], max_norm=1.0)
            optimizer.step()

        # Test each epoch
        print(f"Epoch {epoch}", end='\r')
        with torch.no_grad():
            # Compute training distortion on last batch of training data (X, D, etc.)
            norm_templates = F.normalize(templates, dim=1)
            train_features = shape_max_filter(norm_templates, X)
            train_features = L @ train_features
            DfX_train = torch.cdist(train_features.T, train_features.T)
            alpha_train, beta_train = lipschitz(D, DfX_train)
            distortion_train = (beta_train / alpha_train).item()

            # Compute test distortion ---
            # initalize alpha and beta
            alpha_test = torch.tensor(float("inf"), device=device)
            beta_test  = torch.tensor(0, device=device)

            # Choose full test set
            fX_test = L @ shape_max_filter(norm_templates, X_test)

            # Compute distance matrices
            DX_test = shape_dist_matrix(X_test)          # domain distances
            DfX_test = torch.cdist(fX_test.T, fX_test.T)         # codomain distances

            # Compute Lipschitz constants
            alpha_test, beta_test = lipschitz(DX_test, DfX_test)

            distortion_test = (beta_test / alpha_test).item()

            train_distortions.append(distortion_train)
            test_distortions.append(distortion_test)
        scheduler.step()

    all_trained_distortions.append(train_distortions)
    all_test_distortions.append(test_distortions)
    all_trained_templates.append(norm_templates.detach().cpu().numpy())
    all_trained_Ls.append(L.detach().cpu().numpy())

mean_final_error = np.mean([d[-1] for d in all_test_distortions])
median_final_error = np.median([d[-1] for d in all_test_distortions])
median_final_train_error = np.median([d[-1] for d in all_trained_distortions])

fig = plt.figure(figsize = (15,5))
plt.ylim(top=5)
for distortion_function in all_test_distortions:
    plt.plot(distortion_function, alpha=0.3, c='red')
# for distortion_function in all_trained_distortions:
#     plt.plot(distortion_function, alpha=0.3, c='blue')

textstr = '\n'.join([
    'Shape Space',
    f'Input Data Dimension: {d}',
    f'Embedding Dimension: {target_dim}',
    f'Batch size: {batch_size}',
    f'Grad steps/epoch: {grad_steps_per_epoch}',
    f'Learning rate: {lr}',
    f'Mean Final Distortion: {mean_final_error:.2f}',
    f'Median Final Distortion: {median_final_error:.2f}',
    f'Median Final Training Distortion: {median_final_train_error:.2f}'
])
# Place text box in upper right in axes coords
props = dict(boxstyle='round', alpha=0.1)
plt.gca().text(0.98, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=props)


plt.title("Trained Linear(Max Filter(X)) Model Test Error")
plt.xlabel("Epoch")
plt.ylabel("Distortion")

# Get current time formatted as YYYY-MM-DD_HH-MM
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
# time-based label for now since all params are shown in the figure
plt.savefig(f'./Results/LMF_{timestamp}.png')
np.save('./L_trained.npy', all_trained_Ls[-1])
np.save('./templates_trained.npy', all_trained_templates[-1])
