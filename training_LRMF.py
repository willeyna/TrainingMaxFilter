from groupy import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

######################################################## MODEL PARAMETERS
# number of templates in max filter
t = 8
# currently maintain dimensionality given by max filtering
target_dim = t

batch_size = 128 # Number of randomly generated samples per minibatch
n_trials = 1 # The number of times we will train a new model from scratch
n_epochs = 100 # The number of training epochs for each model
grad_steps_per_epoch = 200 # The number of gradient descent iterations in each training epoch
lr = 5e-3 # learning rate (default is 1e-3 for ADAM)
lr_period = n_epochs # period for cosine annealing
use_double = False
######################################################## LOADING DATA
X_test_np = np.load('X_test.npy')
if np.iscomplexobj(X_test_np):
    X_test = torch.from_numpy(X_test_np).to(torch.complex64).to(device)
else:
    if use_double:
        X_test = torch.from_numpy(X_test_np).to(torch.float64).to(device)
    else:
        X_test = torch.from_numpy(X_test_np).to(torch.float).to(device)
input_shape = X_test.shape[:-1]
n = X_test.shape[-1]
input_dtype = X_test.dtype

# Load in training data if not generating Gaussian samples ##
# train data is also test data here
# train_dataset = torch.utils.data.TensorDataset(X_test.T)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

######################################################## GROUP ACTION
# G is either a finite GroupAction obj or a continuous group name str
# G = GPU_GroupAction(rotations, input_shape[0], device=device, dtype=X_test.dtype, orders=[4])
G = 'orthogonal_2x2'

finite = isinstance(G, GroupAction)
if finite:
    k = G.order
    X_test_orbits = G.get_orbits(X_test)
else:
    X_test_orbits = X_test
    # overwrite max_filter function with the specific continuous versions
    if G == 'shape':
        max_filter = shape_max_filter
    if G == 'phase':
        max_filter = phase_max_filter
    if G == 'orthogonal':
        max_filter = orthogonal_max_filter
    if G == 'orthogonal_2x2':
        max_filter = orthogonal_max_filter_2x2

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
    templates = torch.randn((t, *(input_shape[::-1])), dtype=X_test.dtype, device=device, requires_grad=True)
    # linear layer
    L = torch.normal(0, 1, (target_dim, t), requires_grad=True, device=device, dtype=X_test.real.dtype)

    optimizer = torch.optim.Adam([L], lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, lr_period)

    for epoch in range(n_epochs):
        # if using training data
        # for batch in train_loader:
        # if sampling gaussian
        for step in range(grad_steps_per_epoch):
            optimizer.zero_grad()

            # if sampling gaussian
            X = torch.randn((*input_shape, batch_size), dtype=input_dtype, device=device)
            # if using training data
            # X = batch[0].T

            if finite:
                # full distance matrix for minibatch-- important to keep 'mini'!
                D = G.dist_matrix(X)           # Tensor (n, n)
                X_orbits = G.get_orbits(X)     # Tensor (k, d, n)
            else:
                D = mf_dist_matrix(max_filter, X)
                X_orbits = X

            # Forward pass
            norm_templates = F.normalize(templates, dim=tuple(range(1, templates.dim())))
            filter_features = max_filter(norm_templates, X_orbits)
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
            # Compute training distortion on last batch of training data
            norm_templates = F.normalize(templates, dim=tuple(range(1, templates.dim())))
            train_features = max_filter(norm_templates, X_orbits)
            train_features = L @ train_features
            DfX_train = torch.cdist(train_features.T, train_features.T)
            alpha_train, beta_train = lipschitz(D, DfX_train)
            distortion_train = (beta_train / alpha_train).item()

            fX_test = L @ max_filter(norm_templates, X_test_orbits)
            if finite:
                D_test   = G.dist_matrix(X_test)
            else:
                D_test   = mf_dist_matrix(max_filter, X_test_orbits)

            DfX_test = torch.cdist(fX_test.T, fX_test.T)
            # Compute test distortion
            alpha_test, beta_test = lipschitz(D_test, DfX_test)

            distortion_test = (beta_test / alpha_test).item()

            train_distortions.append(distortion_train)
            test_distortions.append(distortion_test)
        scheduler.step()

    all_trained_distortions.append(train_distortions)
    all_test_distortions.append(test_distortions)
    all_trained_templates.append(norm_templates.detach().cpu().numpy())
    all_trained_Ls.append(L.detach().cpu().numpy())

median_final_train_error = np.median([d[-1] for d in all_trained_distortions])

final_distortions = [d[-1] for d in all_test_distortions]
mean_final_error = np.mean(final_distortions)
median_final_error = np.median(final_distortions)
min_final_error = np.min(final_distortions)

fig = plt.figure(figsize = (15,5))
for distortion_function in all_test_distortions:
    plt.plot(distortion_function, alpha=0.3, c='red')
for distortion_function in all_trained_distortions:
    plt.plot(distortion_function, alpha=0.1, c='blue')

textstr = '\n'.join([
    str(G),
    f'Input Data Dimension: {str(input_shape)}',
    f'Embedding Dimension: {target_dim}',
    f'Batch size: {batch_size}',
    f'Grad steps/epoch: {grad_steps_per_epoch}',
    f'Learning rate: {lr}',
    f'Mean Final Distortion: {mean_final_error:.2f}',
    f'Median Final Distortion: {median_final_error:.2f}',
    f'Best Final Distortion: {min_final_error:.2f}',
    f'Median Final Training Distortion: {median_final_train_error:.2f}'
])
# Place text box in upper right in axes coords
props = dict(boxstyle='round', alpha=0.1)
plt.gca().text(0.98, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=props)


plt.title("Trained Linear(Random Max Filter(X)) Model Test Error")
plt.xlabel("Epoch")
plt.ylabel("Distortion")

# Get current time formatted as YYYY-MM-DD_HH-MM
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
# time-based label for now since all params are shown in the figure
plt.savefig(f'./Results/LRMF_{timestamp}.png')
